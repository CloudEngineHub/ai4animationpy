# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch.nn as nn
from ai4animation.AI.Library import Defaults, Losses
from ai4animation.AI.Library.Blocks import LinearBlock
from ai4animation.AI.Library.Layers import QuantizationLayer
from ai4animation.AI.Library.Statistics import RunningStatistics


class Model(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        embedding_channels,
        embedding_classes,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
        distance="cos",
        anchor="closest",
    ):
        super(Model, self).__init__()

        self.Statistics = RunningStatistics(feature_dim)

        self.Encoder = LinearBlock(
            feature_dim,
            embedding_channels * embedding_classes,
            hidden_dim,
            dropout,
            activation,
        )
        self.Quantizer = QuantizationLayer(
            embedding_channels, embedding_classes, distance=distance, anchor=anchor
        )
        self.Decoder = LinearBlock(
            embedding_channels * embedding_classes,
            feature_dim,
            hidden_dim,
            dropout,
            activation,
        )

    def input_dim(self):
        return self.Encoder.L1.InputSize

    def output_dim(self):
        return self.Encoder.L3.OutputSize

    def feature_dim(self):
        return self.Encoder.L1.InputSize

    def embedding_dim(self):
        return self.Decoder.L1.InputSize

    def forward(self, x):
        x = self.Statistics.Normalize(x)
        z = self.Encoder(x)
        z, _, _ = self.Quantizer(z)
        z = self.Decoder(z)
        y = self.Statistics.Denormalize(z)
        return y

    def learn(self, features, update_statistics):
        if update_statistics:
            self.Statistics.Update(features)

        features = self.Statistics.Normalize(features)
        embedding = self.Encoder(features)
        codes, vq_loss, _ = self.Quantizer(embedding)
        prediction = self.Decoder(codes)
        reconstruction = self.Statistics.Denormalize(prediction)
        return {"Y": reconstruction, "Z": embedding}, {
            "MSE Loss": Losses.MSE(prediction, features),
            "VQ Loss": vq_loss,
        }
