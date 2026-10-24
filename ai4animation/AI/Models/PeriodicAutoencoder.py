# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
import torch.nn as nn
import torch.nn.functional as F
from ai4animation.AI.Library import Defaults, Losses
from ai4animation.AI.Library.Blocks import LinearBlock, SpaceTimeBlock
from ai4animation.AI.Library.Statistics import RunningStatistics


class Model(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        phase_channels,
        sequence_window,
        sequence_length,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
    ):
        super(Model, self).__init__()

        self.Statistics = RunningStatistics(feature_dim)

        self.Channels = phase_channels
        self.SequenceWindow = sequence_window
        self.SequenceLength = sequence_length

        # Nyquist limit for the sampled window (t = linspace(0, sequence_window,
        # seq), so dt = sequence_window / (seq - 1)): frequencies above this alias.
        max_frequency = (sequence_length - 1) / (2.0 * sequence_window)
        print(f"Maximum resolvable frequency: {max_frequency} Hz")

        self.Compressor = SpaceTimeBlock(
            feature_dim,
            hidden_dim,
            hidden_dim,
            sequence_length,
            1,
            dropout,
        )
        self.PhaseEncoder = LinearBlock(
            hidden_dim,
            phase_channels * 2,
            hidden_dim,
            dropout,
            activation,
        )
        self.FrequencyEncoder = LinearBlock(
            hidden_dim,
            phase_channels,
            hidden_dim,
            dropout,
            activation,
        )
        self.AmplitudeEncoder = LinearBlock(
            hidden_dim,
            phase_channels,
            hidden_dim,
            dropout,
            activation,
        )
        self.Decoder = LinearBlock(
            phase_channels * 2, feature_dim, hidden_dim, dropout, activation
        )

    def input_dim(self):
        return self.Compressor.input_dim()

    def output_dim(self):
        return self.Compressor.output_dim()

    def feature_dim(self):
        return self.Compressor.input_dim()

    def embedding_dim(self):
        return self.Decoder.L1.InputSize

    def forward(self, x):
        x = self.Statistics.Normalize(x)
        embedding = self.Compressor(x).squeeze(-2)  # (B, hidden_dim)
        m = self.transform(
            self.PhaseEncoder(embedding),
            self.FrequencyEncoder(embedding),
            self.AmplitudeEncoder(embedding),
        )
        z = self.Decoder(m.reshape(*m.shape[:-2], self.Channels * 2))
        y = self.Statistics.Denormalize(z)
        return y, m

    def learn(self, features, update_statistics):
        if update_statistics:
            self.Statistics.Update(features)

        features = self.Statistics.Normalize(features)

        embedding = self.Compressor(features).squeeze(-2)  # (B, hidden_dim)
        manifold = self.transform(
            self.PhaseEncoder(embedding),
            self.FrequencyEncoder(embedding),
            self.AmplitudeEncoder(embedding),
        )
        prediction = self.Decoder(
            manifold.reshape(*manifold.shape[:-2], self.Channels * 2)
        )

        reconstruction_loss = Losses.MSE(prediction, features)

        reconstruction = self.Statistics.Denormalize(prediction)
        return {"Y": reconstruction, "M": manifold}, {
            "MSE Loss": reconstruction_loss,
        }

    def transform(self, phase, frequency, amplitude):
        # phase:        (B, 2C) phase vector per channel at the center frame.
        # frequency:    (B, C) single frequency (Hz) per channel for the window.
        # amplitude:    (B, C) single amplitude per channel for the window.

        # Normalize before atan2 so a near-zero vector can't produce NaN gradients.
        phase = F.normalize(phase.reshape(-1, self.Channels, 2), dim=-1, eps=1e-8)
        # Compute phase angle
        phase = -torch.atan2(phase[..., 1], phase[..., 0]) / (
            2.0 * torch.pi
        )  # (B, C), in cycles

        # Non-negative frequency (Hz). Phase advances linearly in physical time from
        # the predicted center phase: t is measured in seconds relative to the center
        # frame, so the center frame keeps exactly the predicted phase.
        frequency = F.softplus(frequency)  # >= 0 Hz, (B, C)

        # Amplitude scales each phase vector's magnitude, constrained to [0, 1].
        amplitude = torch.sigmoid(amplitude)  # in [0, 1], (B, C)

        # Timestamps
        t = torch.linspace(
            -self.SequenceWindow / 2.0,
            self.SequenceWindow / 2.0,
            self.SequenceLength,
            device=phase.device,
        ).reshape(1, self.SequenceLength, 1)  # (1, S, 1), seconds

        # Compute phases across the window
        sequence = phase.unsqueeze(1) + frequency.unsqueeze(1) * t  # (B, S, C), cycles

        # Negated sine => clockwise forward rotation.
        angle = 2.0 * torch.pi * sequence  # (B, S, C)
        manifold = torch.stack([torch.cos(angle), -torch.sin(angle)], dim=-1)
        # Scale unit phase vectors by per-channel amplitude in [0, 1].
        manifold = amplitude.reshape(-1, 1, self.Channels, 1) * manifold
        return manifold
