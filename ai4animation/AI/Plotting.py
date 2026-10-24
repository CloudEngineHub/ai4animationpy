# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Plotting utilities for visualizing latent spaces and training metrics."""

import warnings
from collections import deque

import matplotlib.pyplot as plt
from ai4animation.AI.Library.Statistics import WelfordStatistics
from sklearn.decomposition import PCA


def CreateFigure(title=None, figsize=None):
    plt.ion()
    figure = plt.figure(figsize=figsize)
    if title is not None:
        figure.canvas.manager.set_window_title(title)
    plt.show(block=False)
    return figure


def RefreshFigure(figure, tight_layout=False):
    if tight_layout:
        figure.tight_layout()
    figure.canvas.draw_idle()
    figure.canvas.flush_events()


def PCA2D(ax, values, title):  # values is batch x seq x dim
    x = ToNumpy(values)

    batch = x.shape[0]
    dim = x.shape[1]

    ax.cla()

    point_alpha = 0.3
    line_alpha = 0.2

    y = PCA(n_components=2).fit_transform(x.reshape(-1, dim)).reshape(batch, 2)
    for i in range(y.shape[0]):
        px = y[i, 0]
        py = y[i, 1]
        ax.plot(px, py, c=(0, 0, 0), alpha=line_alpha)
        ax.scatter(px, py, alpha=point_alpha)

    ax.set_axis_off()
    ax.set_title(title)


def PCA2DSequence(ax, values, title):  # values is batch x seq x dim
    x = ToNumpy(values)

    batch = x.shape[0]
    seq = x.shape[1]
    dim = x.shape[2]

    ax.cla()

    point_alpha = 0.3
    line_alpha = 0.2

    y = PCA(n_components=2).fit_transform(x.reshape(-1, dim)).reshape(batch, seq, 2)
    for i in range(y.shape[0]):
        px = y[i, :, 0]
        py = y[i, :, 1]
        ax.plot(px, py, c=(0, 0, 0), alpha=line_alpha)
        ax.scatter(px, py, alpha=point_alpha)

    ax.set_axis_off()
    ax.set_title(title)


def PCA3DSequence(ax, values, title):  # values is batch x seq x dim
    x = ToNumpy(values)

    batch = x.shape[0]
    seq = x.shape[1]
    dim = x.shape[2]

    ax.cla()

    point_alpha = 0.3
    line_alpha = 0.2

    y = PCA(n_components=3).fit_transform(x.reshape(-1, dim)).reshape(batch, seq, 3)
    for i in range(y.shape[0]):
        px = y[i, :, 0]
        py = y[i, :, 1]
        pz = y[i, :, 2]
        ax.plot(px, py, pz, c=(0, 0, 0), alpha=line_alpha)
        ax.scatter(px, py, pz, alpha=point_alpha)

    limit = 3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_axis_off()
    ax.set_title(title)


def PlotFunctions(ax, values, title, step=1, yLimits=None):
    values = ToNumpy(values)

    ax.cla()
    for i in range(0, len(values), step):
        ax.plot(values[i])
    if yLimits is not None:
        ax.set_ylim([yLimits[0], yLimits[1]])
    ax.set_axis_off()
    ax.set_title(title)


def PlotScatter(ax, values, title, step=1, xLimits=None, yLimits=None, labels=None):
    values = ToNumpy(values)
    labels = ToNumpy(labels)

    ax.cla()
    # ax.scatter(values[..., 0], values[..., 1])
    ax.scatter(values[:, 0], values[:, 1], c=labels, cmap="viridis", edgecolors="k")
    if xLimits is not None:
        ax.set_xlim([xLimits[0], xLimits[1]])
    if yLimits is not None:
        ax.set_ylim([yLimits[0], yLimits[1]])
    ax.set_axis_off()
    ax.set_title(title)


def PlotGridMap(ax, values, title):
    ax.cla()
    ax.imshow(values, cmap=plt.cm.bone, aspect="auto")
    ax.set_axis_off()
    ax.set_title(title)


def ToNumpy(value):
    return value.detach().cpu().numpy()


class LossHistory:
    def __init__(
        self,
        title,
        horizon,  # None means forever
        ax=None,
        min=None,
        max=None,
        cumulativeHorizon=100,
        drawInterval=100,
        drawWait=1e-3,
        yScale="linear",
    ):
        warnings.filterwarnings(
            "ignore",
            message="Attempt to set non-positive ylim on a log-scaled axis will be ignored.",
        )
        if ax is None and drawInterval is not None:
            self.figure = CreateFigure(title, figsize=(10, 5))
            self.ax = self.figure.subplots()
        else:
            self.ax = ax
            self.figure = ax.figure if ax is not None else None
        self.Title = title
        self.Horizon = horizon
        self.CumulativeHorizon = cumulativeHorizon
        self.DrawInterval = drawInterval
        self.DrawWait = drawWait
        self.YMin = min
        self.YMax = max
        self.YRange = [0 if min == None else min, 0 if max == None else max]
        self.Functions = {}  # string->[History, Horizon]
        self.Running = {}  # string->WelfordStatistics
        self.Counter = 0
        self.YScale = yScale

    def Close(self):
        if self.figure is not None:
            plt.close(self.figure)

    def Add(self, dict):  # Dictionary contains tuples of (value, label)
        if self.DrawInterval is None:
            for name, value in dict.items():
                if name not in self.Running:
                    self.Running[name] = WelfordStatistics()
                self.Running[name].Update(value)
            return
        for k, v in dict.items():
            value = ToNumpy(v)
            label = k
            if label not in self.Functions:
                maxlen = self.Horizon if self.Horizon != 0 else None
                self.Functions[label] = (
                    deque(maxlen=maxlen),
                    deque(maxlen=maxlen),
                    deque(maxlen=self.CumulativeHorizon),  # rolling window for avg
                )
            function = self.Functions[label]
            function[0].append(value)
            function[2].append(value)
            cumulative = sum(function[2]) / len(function[2])
            function[1].append(cumulative)

            self.YRange[0] = (
                min(self.YRange[0], 0.5 * cumulative)
                if self.YMin == None
                else self.YRange[0]
            )
            self.YRange[1] = (
                max(self.YRange[1], 2 * cumulative)
                if self.YMax == None
                else self.YRange[1]
            )

        if self.DrawInterval is not None:
            self.Counter += 1
            if self.Counter >= self.DrawInterval:
                self.Counter = 0
                self.Draw()

    def Draw(self):
        if self.DrawInterval is None:
            return
        self.ax.cla()
        self.ax.set_title(self.Title)
        for label in self.Functions.keys():
            function = self.Functions[label]
            step = max(int(len(function[0]) / self.DrawInterval), 1)
            values = list(function[0])
            cumulative = list(function[1])
            self.ax.plot(
                values[::step],
                label=label + " (" + str(round(self.CumulativeValue(label), 3)) + ")",
            )
            self.ax.plot(cumulative[::step], c=(0, 0, 0))
        self.ax.set_yscale(self.YScale)
        self.ax.set_ylim((self.YRange[0], self.YRange[1]))
        self.ax.legend()
        self.ax.set_axis_off()
        RefreshFigure(self.figure)

    def Value(self, label=None):
        if label == None:
            return sum(x[0][-1] for x in self.Functions.values())
        else:
            return self.Functions[label][0][-1]

    def CumulativeValue(self, label=None):
        if label == None:
            return sum(x[1][-1] for x in self.Functions.values())
        else:
            return self.Functions[label][1][-1]

    def Print(self, digits=5, reset=True, end="\n"):
        if self.DrawInterval is None:
            output = ""
            for name, statistics in self.Running.items():
                mean = ToNumpy(statistics.GetMean()).item()
                std = ToNumpy(statistics.GetStd()).item()
                output += (
                    name
                    + ": "
                    + str(round(mean, digits))
                    + " (std: "
                    + str(round(std, digits))
                    + ") "
                )
            print(output, end=end, flush=True)
            if reset:
                self.Running.clear()
            return
        output = ""
        for name in self.Functions.keys():
            output = (
                output
                + name
                + ": "
                + str(round(self.CumulativeValue(name), digits))
                + " "
            )
        print(output)
