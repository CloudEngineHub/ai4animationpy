# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
import torch.nn as nn
from ai4animation.AI.Library import Losses
from ai4animation.AI.Library.Blocks import FiLMLinearBlock, LinearBlock, SpaceTimeBlock
from ai4animation.AI.Library.Layers import CodebookLayer
from ai4animation.AI.Library.Statistics import RunningStatistics
from ai4animation.AI.Models import MultiLayerPerceptron


# Codebook Matching Model
class Model(nn.Module):
    def __init__(
        self,
        prior,
        sampler,
    ):
        super(Model, self).__init__()
        self.Prior = prior
        self.Sampler = sampler

    def input_dim(self):
        return self.Sampler.input_dim()

    def output_dim(self):
        return self.Prior.Decoder.output_dim()

    def forward(self, x, iterations=1, sample=False, results=None):
        if sample is False:
            noise = torch.zeros(x.shape[0], self.Sampler.Codebook.Size, device=x.device)
        elif sample is True:
            noise = torch.rand(x.shape[0], self.Sampler.Codebook.Size, device=x.device)
        else:
            noise = sample

        if results is None:
            return self.Prior(self.Sampler(x, iterations, noise))
        else:
            z, r = self.Sampler(x, iterations, noise, [])
            for i in range(len(r)):
                r[i] = self.Prior(r[i])
            y = self.Prior(z)
            return y, r

    def learn(self, input, output, update_statistics, detach_gradients=False):
        prior_result, prior_loss = self.Prior.learn(output, update_statistics)
        sampler_targets = prior_result["Z"]
        if detach_gradients:
            sampler_targets = sampler_targets.detach()
        sampler_result, sampler_loss = self.Sampler.learn(
            input, sampler_targets, update_statistics
        )
        result = prior_result | sampler_result
        loss = prior_loss | sampler_loss
        return result, loss


# Codebook Matching Prior
class Prior(nn.Module):
    def __init__(self, encoder, decoder, codebook):
        super(Prior, self).__init__()
        self.Encoder = encoder
        self.Decoder = decoder
        self.Codebook = codebook
        self.FeatureStatistics = RunningStatistics(decoder.output_dim())

    def forward(self, codes):
        y = self.decode(codes)
        y = self.FeatureStatistics.Denormalize(y)
        return y

    def target(self, y):
        y = self.FeatureStatistics.Normalize(y)
        z = self.encode(y)
        return z

    def encode(self, x):
        return self.Encoder(x)

    def decode(self, x):
        return self.Decoder(x)

    def reconstruct(self, y, sample):
        y = self.FeatureStatistics.Normalize(y)
        z = self.encode(y)
        c = self.Codebook(z, sample=sample)
        y = self.decode(c)
        y = self.FeatureStatistics.Denormalize(y)
        return y

    def learn(self, features, update_statistics):
        if update_statistics:
            self.FeatureStatistics.Update(features)
        features = self.FeatureStatistics.Normalize(features)
        logits = self.encode(features)
        codes = self.Codebook(logits, sample=True)
        probs = self.Codebook(logits, sample=False)

        # # probs: [B, Channels * Classes]  -- per-sample softmax already from Codebook
        # B = probs.shape[0]
        # p = probs.reshape(B, self.Codebook.Channels, self.Codebook.Classes)
        # eps = 1e-8

        # # 1) Coverage: average over batch should be uniform per channel
        # avg_probs = p.mean(dim=0)                                  # [Ch, Cl]
        # uniform   = torch.full_like(avg_probs, 1.0 / self.Codebook.Classes)
        # # KL(avg || uniform) == log(K) - H(avg). Minimizing == maximizing H(avg).
        # coverage_loss = (avg_probs * (avg_probs.add(eps).log() -
        #                             uniform.log())).sum(-1).mean()

        # # 2) Sharpness: each sample should commit to one class per channel
        # sample_entropy = -(p * p.add(eps).log()).sum(-1).mean()  # mean over B, Ch

        pred = self.decode(codes)
        result = {
            "Z": logits,
            "P": probs,
            "C": codes,
            "Y": self.FeatureStatistics.Denormalize(pred),
        }
        loss = {
            "Reconstruction Loss": Losses.MSE(pred, features),
            # "Coverage Loss":  0.1  * coverage_loss,
            # "Sharpness Loss": 0.01 * sample_entropy,
        }
        return result, loss


# Codebook Matching Sampler
class Sampler(nn.Module):
    def __init__(
        self,
        feature_dim,
        estimator_dim,
        codebook_channels,
        codebook_classes,
        dropout,
        learn_denoiser=True,
    ):
        super(Sampler, self).__init__()
        self.Projector = LinearBlock(
            feature_dim,
            codebook_channels * codebook_classes,
            estimator_dim,
            dropout,
        )
        self.Denoiser = (
            LinearBlock(
                codebook_channels * codebook_classes + feature_dim,
                codebook_channels * codebook_classes,
                estimator_dim,
                dropout,
            )
            if learn_denoiser
            else None
        )
        self.Codebook = CodebookLayer(codebook_channels, codebook_classes)
        self.FeatureStatistics = RunningStatistics(self.input_dim())

    def input_dim(self):
        return self.Projector.input_dim()

    def forward(self, x, iterations, noise, results=None):
        x = self.FeatureStatistics.Normalize(x)

        c = self.Codebook(self.project(x), sample=noise)
        if results is not None:
            results.append(c.clone())

        if self.Denoiser is not None:
            for _ in range(iterations):
                c = self.Codebook(self.denoise(c, x), sample=False)
                if results is not None:
                    results.append(c.clone())
        # elif iterations > 0:
        #     print("Denoising is disabled for this sampler")

        if results is not None:
            return c, results
        else:
            return c

    def project(self, x):
        return self.Projector(x)

    def denoise(self, z, x):
        if self.Denoiser is None:
            raise RuntimeError("Denoising is disabled for this sampler")
        return self.Denoiser(torch.cat((z, x), -1))

    def normalize(self, N):
        N = N.reshape(-1, self.Codebook.Classes)
        N = N / torch.sum(N, axis=-1).reshape(-1, 1)
        N = N.reshape(-1, self.Codebook.Size)
        return N

    def learn(self, features, targets, update_statistics):
        if update_statistics:
            self.FeatureStatistics.Update(features)
        features = self.FeatureStatistics.Normalize(features)

        projection = self.project(features)
        target = self.Codebook(targets, sample=False)

        # Project
        estimate = self.Codebook(projection, sample=False)
        result = {"P": estimate}
        loss = {"Projection Loss": Losses.MSE(estimate, target)}

        # Denoise
        if self.Denoiser is not None:
            U = torch.rand(targets.shape[0], 1, device=features.device)
            source = self.Codebook(projection, sample=True)
            seed = self.normalize((1 - U) * target + U * source)
            denoised = self.Codebook(self.denoise(seed, features), sample=False)
            result["N"] = denoised
            loss["Denoising Loss"] = Losses.MSE(denoised, target)

        return result, loss


####################################################################################################
####################################################################################################
####################################################################################################
# Priors
####################################################################################################
####################################################################################################
####################################################################################################
class DensePrior(Prior):
    def __init__(
        self,
        feature_dim,
        encoder_dim,
        decoder_dim,
        codebook_channels,
        codebook_classes,
        dropout,
    ):
        super(DensePrior, self).__init__(
            encoder=LinearBlock(
                feature_dim,
                codebook_channels * codebook_classes,
                encoder_dim,
                dropout,
            ),
            decoder=LinearBlock(
                codebook_channels * codebook_classes,
                feature_dim,
                decoder_dim,
                dropout,
            ),
            codebook=CodebookLayer(codebook_channels, codebook_classes),
        )


class SequentialPrior(Prior):
    def __init__(
        self,
        sequence_length,
        feature_dim,
        encoder_dim,
        decoder_dim,
        codebook_channels,
        codebook_classes,
        dropout,
    ):
        super(SequentialPrior, self).__init__(
            encoder=SpaceTimeBlock(
                feature_dim,
                codebook_channels * codebook_classes,
                encoder_dim,
                sequence_length,
                1,
                dropout,
            ),
            decoder=FiLMLinearBlock(
                codebook_channels * codebook_classes,
                feature_dim,
                decoder_dim,
                1,
                dropout,
            ),
            codebook=CodebookLayer(codebook_channels, codebook_classes),
        )

        self.SequenceLength = sequence_length

        self.TimeStatistics = RunningStatistics(1)
        for _ in range(100):
            self.TimeStatistics.Update(self.timing())

    def timing(self, device=None):
        if device is not None:
            t = torch.linspace(0.0, 1.0, self.SequenceLength, device=device)
        else:
            t = torch.linspace(0.0, 1.0, self.SequenceLength)
        return t.reshape(-1, 1)

    def encode(self, x):
        return self.Encoder(x).squeeze(-2)

    def decode(self, x):
        x = x.unsqueeze(1).repeat(1, self.SequenceLength, 1)
        t = self.TimeStatistics.Normalize(self.timing(x.device))
        t = t.reshape(1, -1, 1).repeat(x.shape[0], 1, 1)
        return self.Decoder(x, t)


####################################################################################################
####################################################################################################
####################################################################################################
# Models
####################################################################################################
####################################################################################################
####################################################################################################
# Input: [Batch, InputDim]
# Output: [Batch, OutputDim]
class DenseModel(Model):
    def __init__(
        self,
        input_dim,
        output_dim,
        encoder_dim,
        decoder_dim,
        estimator_dim,
        codebook_channels,
        codebook_classes,
        dropout,
        learn_denoiser=True,
    ):
        super(DenseModel, self).__init__(
            prior=DensePrior(
                output_dim,
                encoder_dim,
                decoder_dim,
                codebook_channels,
                codebook_classes,
                dropout,
            ),
            sampler=Sampler(
                input_dim,
                estimator_dim,
                codebook_channels,
                codebook_classes,
                dropout,
                learn_denoiser=learn_denoiser,
            ),
        )


# Input: [Batch, InputDim]
# Output: [Batch, Sequence, OutputDim]
class SequentialModel(Model):
    def __init__(
        self,
        sequence_length,
        input_dim,
        output_dim,
        encoder_dim,
        decoder_dim,
        estimator_dim,
        codebook_channels,
        codebook_classes,
        dropout,
        learn_denoiser=True,
    ):
        super(SequentialModel, self).__init__(
            prior=SequentialPrior(
                sequence_length,
                output_dim,
                encoder_dim,
                decoder_dim,
                codebook_channels,
                codebook_classes,
                dropout,
            ),
            sampler=Sampler(
                input_dim,
                estimator_dim,
                codebook_channels,
                codebook_classes,
                dropout,
                learn_denoiser=learn_denoiser,
            ),
        )


####################################################################################################
####################################################################################################
####################################################################################################
# Toy Example Ambiguous Square Functions
####################################################################################################
####################################################################################################
####################################################################################################
def ToyExample1():
    FEATURES = 10
    SEQUENCE_LENGTH = 1
    EPOCH_COUNT = 150
    BATCH_COUNT = 1000
    BATCH_SIZE = 32
    ITERATIONS = 10
    TEST_SIZE = 128
    DRAW_INTERVAL = 100

    model = DenseModel(
        input_dim=1,
        output_dim=FEATURES,
        encoder_dim=64,
        decoder_dim=64,
        estimator_dim=64,
        codebook_channels=16,
        codebook_classes=8,
        dropout=0.0,
    )

    # model = SequentialModel(
    #     sequence_length=SEQUENCE_LENGTH,
    #     input_dim=1,
    #     output_dim=FEATURES,
    #     encoder_dim=64,
    #     decoder_dim=64,
    #     estimator_dim=64,
    #     codebook_channels=16,
    #     codebook_classes=8,
    #     dropout=0.0,
    # )

    optimizer = torch.optim.Adam(model.parameters())

    loss_history = Plotting.LossHistory(
        "Loss History", BATCH_COUNT, drawInterval=DRAW_INTERVAL, yScale="log"
    )

    def generate_data(batch, min, max):
        X1, Y1 = Generators.AmbiguousSquareFunctions(
            int(batch / 2), FEATURES * SEQUENCE_LENGTH, min, max
        )
        X2, Y2 = Generators.AmbiguousSineFunctions(
            int(batch / 2), FEATURES * SEQUENCE_LENGTH, min, max
        )
        X = torch.cat((X1, X2), dim=0)
        Y = torch.cat((Y1, Y2), dim=0)
        # Y = Y.reshape(-1, SEQUENCE_LENGTH, FEATURES)
        return (X, Y)

    min = 1.0
    max = 4.0
    figure = Plotting.CreateFigure("Codebook Matching Functions", figsize=(9, 3))
    axes = figure.subplots(1, 3)
    for e in range(1, EPOCH_COUNT + 1, 1):
        print("Epoch", e)
        for i in range(0, BATCH_COUNT):
            print("Progress", round(100 * i / BATCH_COUNT, 2), "%", end="\r")

            X, Y = generate_data(BATCH_SIZE, min, max)

            _, loss = model.learn(X, Y, e == 1)

            optimizer.zero_grad()
            sum(loss.values()).backward()
            optimizer.step()

            loss_history.Add(loss)

            # PLOTTING
            if i % DRAW_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    Xtrue, Ytrue = generate_data(TEST_SIZE, min, max)
                    Ypred = model(
                        Xtrue,
                        iterations=ITERATIONS,
                        sample=True,
                    )
                    Yrec = model.Prior.reconstruct(Ytrue, sample=True)

                    Plotting.PlotFunctions(
                        axes[0],
                        Ytrue.detach().flatten(start_dim=1),
                        "Ground Truth",
                        step=1,
                        yLimits=[-5, 5],
                    )

                    Plotting.PlotFunctions(
                        axes[1],
                        Yrec.detach().flatten(start_dim=1),
                        "Reconstruction",
                        step=1,
                        yLimits=[-5, 5],
                    )

                    Plotting.PlotFunctions(
                        axes[2],
                        Ypred.detach().flatten(start_dim=1),
                        "Sampling",
                        step=1,
                        yLimits=[-5, 5],
                    )

                    # fig, axes = plots_steps
                    # for s in range(len(steepness)):
                    #     for i in range(ITERATIONS+1):
                    #         y = model(
                    #             steepness[s] * torch.ones_like(Xtrue),
                    #             iterations=i,
                    #             sample=True,
                    #         )
                    #         Plotting.PlotFunctions(
                    #             axes[s, i],
                    #             y.detach().flatten(start_dim=1),
                    #             "X=" + str(steepness[s]) + " I=" + str(i),
                    #             step=1,
                    #             yLimits=[-5, 5],
                    #         )

                model.train()
                Plotting.RefreshFigure(figure, tight_layout=True)


####################################################################################################
####################################################################################################
####################################################################################################
# Toy Example Four Moons
####################################################################################################
####################################################################################################
####################################################################################################
def ToyExample2():
    INPUT_DIM = 1
    OUTPUT_DIM = 2
    EPOCH_COUNT = 150
    BATCH_COUNT = 1000
    BATCH_SIZE = 256
    ITERATIONS = 8
    TEST_SIZE = 1024
    DRAW_INTERVAL = 100

    model = DenseModel(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        encoder_dim=128,
        decoder_dim=128,
        estimator_dim=128,
        codebook_channels=32,
        codebook_classes=4,
        dropout=0.0,
    )

    optimizer = torch.optim.Adam(model.parameters())

    loss_history = Plotting.LossHistory(
        "Loss History", BATCH_COUNT, drawInterval=DRAW_INTERVAL, yScale="log"
    )

    def generate_data(batch, noise):
        Y, X = Generators.FourMoons(batch, noise)
        X = X.reshape(-1, 1)
        return (X, Y)

    min = -1.25
    max = 2.5
    noise = 0.05
    height = 3.0
    width = 2.0
    figure = Plotting.CreateFigure(
        "Codebook Matching Four Moons",
        figsize=((3 + ITERATIONS) * width, height),
    )
    axes = figure.subplots(1, 3 + ITERATIONS)
    for e in range(1, EPOCH_COUNT + 1, 1):
        print("Epoch", e)
        for i in range(0, BATCH_COUNT):
            print("Progress", round(100 * i / BATCH_COUNT, 2), "%", end="\r")

            X, Y = generate_data(BATCH_SIZE, noise)

            _, loss = model.learn(X, Y, e == 1)

            optimizer.zero_grad()
            sum(loss.values()).backward()
            optimizer.step()

            loss_history.Add(loss)

            # PLOTTING
            if i % DRAW_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    Xtrue, Ytrue = generate_data(TEST_SIZE, noise)
                    Ypred, results = model(
                        Xtrue,
                        iterations=ITERATIONS,
                        sample=True,
                        results=True,
                    )
                    Yrec = model.Prior.reconstruct(Ytrue, sample=True)

                    Plotting.PlotScatter(
                        axes[0],
                        Ytrue.detach().reshape(-1, 2),
                        "Ground Truth",
                        step=1,
                        yLimits=[min, max],
                        labels=Xtrue.detach().reshape(-1, 1),
                    )

                    Plotting.PlotScatter(
                        axes[1],
                        Yrec.detach().reshape(-1, 2),
                        "Reconstruction",
                        step=1,
                        yLimits=[min, max],
                        labels=Xtrue.detach().reshape(-1, 1),
                    )

                    for k in range(len(results)):
                        Plotting.PlotScatter(
                            axes[2 + k],
                            results[k].detach().reshape(-1, 2),
                            "Sampling",
                            step=1,
                            yLimits=[min, max],
                            labels=Xtrue.detach().reshape(-1, 1),
                        )

                model.train()
                Plotting.RefreshFigure(figure, tight_layout=True)


if __name__ == "__main__":
    from ai4animation import Generators, Plotting

    # ToyExample1()
    ToyExample2()
