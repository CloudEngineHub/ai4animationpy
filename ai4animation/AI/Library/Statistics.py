# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class WelfordStatistics:
    """Tracks tensor mean and standard deviation online without retaining samples."""

    def __init__(self):
        self.clear()

    def Update(self, value):
        value = value.detach()
        if self.Count == 0:
            self.Count = 1
            self.Mean = value.clone()
            self.SquaredDeviations = value.new_zeros(value.shape)
            return
        if value.shape != self.Mean.shape:
            raise ValueError("value shape must match existing statistics")

        self.Count += 1
        delta = value - self.Mean
        self.Mean.add_(delta / self.Count)
        self.SquaredDeviations.add_(delta * (value - self.Mean))

    def GetMean(self):
        if self.Count == 0:
            raise RuntimeError("statistics have not been updated")
        return self.Mean

    def GetVariance(self):
        if self.Count == 0:
            raise RuntimeError("statistics have not been updated")
        return self.SquaredDeviations / self.Count

    def GetStd(self):
        return self.GetVariance().sqrt()

    def clear(self):
        self.Count = 0
        self.Mean = None
        self.SquaredDeviations = None


class RunningStatistics(nn.Module):
    """Computes running mean and standard deviation
    Url: https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501
    Adapted from:
        *
        <http://stackoverflow.com/questions/1174984/how-to-efficiently-\
calculate-a-running-standard-deviation>
        * <http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html>
        * <https://gist.github.com/fvisin/5a10066258e43cf6acfa0a474fcdb59f>

    Adapted for usage with PyTorch by converting the original implementation from NumPy to PyTorch.
    """

    def __init__(self, dim, n=0.0, m=None, s=None, id=None):
        super(RunningStatistics, self).__init__()

        self.Active = False  # Flag to check if the statistics have been updated

        self.n = n
        self.m = m
        self.s = s
        self.ID = id

        self.Dim = dim
        self.StdGroups = []

        self.Norm = Parameter(
            torch.from_numpy(
                np.vstack(
                    (np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32))
                )
            ),
            requires_grad=False,
        )

    def AddGroup(self, group):
        indices = list(group)
        if not indices:
            raise ValueError("std groups must not be empty")

        groups = getattr(self, "StdGroups", None) or []
        used = {index for existing in groups for index in existing}
        normalized = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                raise ValueError("std group indices must be integers")
            index = int(index)
            if index < 0 or index >= self.Dim:
                raise ValueError(f"std group index {index} is out of range")
            if index in used or index in normalized:
                raise ValueError(f"std group index {index} occurs more than once")
            normalized.append(index)
        groups.append(normalized)
        self.StdGroups = groups

    def Update(self, data):
        values = data.reshape(-1, self.Dim)
        if values.shape[0] == 0:
            return
        self.Active = True
        if data.is_cuda:
            values = values.cpu()
        values = values.detach().numpy()
        batch_n = values.shape[0]
        batch_mean = np.mean(values, axis=0)
        batch_m2 = np.sum((values - batch_mean) ** 2, axis=0)
        if self.m is None:
            self.n = batch_n
            self.m = batch_mean
            self.s = batch_m2
        else:
            new_n = self.n + batch_n
            delta = batch_mean - self.m
            self.m += delta * batch_n / new_n
            self.s += batch_m2 + delta**2 * self.n * batch_n / new_n
            self.n = new_n
        mean = self.mean
        std = self.std
        std[std < 0.001] = 1.0
        for group in getattr(self, "StdGroups", None) or []:
            std[group] = np.mean(std[group])
        self.Norm[0] = torch.from_numpy(mean)
        self.Norm[1] = torch.from_numpy(std)

    def Normalize(self, tensor):
        if not self.Active:
            return tensor
        if tensor.shape[-1] != self.Dim:
            # print(self.ID)
            print(
                "Dimension mismatch in Normalize. Given:",
                tensor.shape[-1],
                " Expected:",
                self.Dim,
            )
            return tensor
        mean = self.Norm[0]
        std = self.Norm[1]
        return (tensor - mean) / std

    def Denormalize(self, tensor):
        if not self.Active:
            return tensor
        if tensor.shape[-1] != self.Dim:
            print(
                "Dimension mismatch in Denormalize. Given:",
                tensor.shape[-1],
                " Expected:",
                self.Dim,
            )
            return tensor
        mean = self.Norm[0]
        std = self.Norm[1]
        return (tensor * std) + mean

    def GetMean(self):
        return self.Norm[0]

    def GetStd(self):
        return self.Norm[1]

    def update_params(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
            self.s = np.zeros_like(x)
        else:
            prev_m = self.m.copy()
            self.m += (x - self.m) / self.n
            self.s += (x - prev_m) * (x - self.m)

    def clear(self):
        self.n = 0.0
        self.m = None
        self.s = None
        self.Active = False

    @property
    def mean(self):
        return self.m

    @property
    def variance(self):
        return self.s / (self.n - 1) if self.n > 1 else np.zeros_like(self.s)

    @property
    def std(self):
        return np.sqrt(self.variance)


# # Copyright (c) Meta Platforms, Inc. and affiliates.
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn.parameter import Parameter


# class RunningStatistics(nn.Module):
#     """Computes running mean and standard deviation
#     Url: https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501
#     Adapted from:
#         *
#         <http://stackoverflow.com/questions/1174984/how-to-efficiently-\
# calculate-a-running-standard-deviation>
#         * <http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html>
#         * <https://gist.github.com/fvisin/5a10066258e43cf6acfa0a474fcdb59f>

#     Adapted for usage with PyTorch by converting the original implementation from NumPy to PyTorch.
#     """

#     def __init__(self, dim, n=0.0, m=None, s=None, id=None):
#         super(RunningStatistics, self).__init__()

#         self.Active = False  # Flag to check if the statistics have been updated

#         self.n = n
#         self.m = m
#         self.s = s
#         self.ID = id

#         self.Dim = dim

#         self.Norm = Parameter(
#             torch.from_numpy(
#                 np.vstack(
#                     (np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32))
#                 )
#             ),
#             requires_grad=False,
#         )

#     def Update(self, data):
#         self.Active = True
#         values = data.reshape(-1, self.Dim)
#         if data.is_cuda:
#             values = values.cpu()
#         values = values.detach().numpy()
#         for k in range(values.shape[0]):
#             self.update_params(values[k])
#         mean = self.mean
#         std = self.std
#         std[std < 0.001] = 1.0
#         self.Norm[0] = torch.from_numpy(mean)
#         self.Norm[1] = torch.from_numpy(std)

#     def Normalize(self, tensor):
#         if not self.Active:
#             return tensor
#         if tensor.shape[-1] != self.Dim:
#             # print(self.ID)
#             print(
#                 "Dimension mismatch in Normalize. Given:",
#                 tensor.shape[-1],
#                 " Expected:",
#                 self.Dim,
#             )
#             return tensor
#         mean = self.Norm[0]
#         std = self.Norm[1]
#         return (tensor - mean) / std

#     def Denormalize(self, tensor):
#         if not self.Active:
#             return tensor
#         if tensor.shape[-1] != self.Dim:
#             print(
#                 "Dimension mismatch in Denormalize. Given:",
#                 tensor.shape[-1],
#                 " Expected:",
#                 self.Dim,
#             )
#             return tensor
#         mean = self.Norm[0]
#         std = self.Norm[1]
#         return (tensor * std) + mean

#     def GetMean(self):
#         return self.Norm[0]

#     def GetStd(self):
#         return self.Norm[1]

#     def update_params(self, x):
#         self.n += 1
#         if self.n == 1:
#             self.m = x
#             self.s = np.zeros_like(x)
#         else:
#             prev_m = self.m.copy()
#             self.m += (x - self.m) / self.n
#             self.s += (x - prev_m) * (x - self.m)

#     def clear(self):
#         self.n = 0.0

#     @property
#     def mean(self):
#         return self.m

#     def variance(self):
#         return self.s / (self.n - 1) if self.n > 1 else np.zeros_like(self.s)

#     @property
#     def std(self):
#         return np.sqrt(self.variance())
