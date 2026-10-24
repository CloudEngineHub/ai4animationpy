# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Dataset loader for NPZ motion capture files."""

import os

from ai4animation.Animation.Motion import Motion


class Dataset:
    class Instance:
        __slots__ = ("Motion", "Index")

        def __init__(self, motion, index):
            self.Motion = motion
            self.Index = index

    def __init__(self, directory, modules=None, operation=None, max_files=None):
        self.Directory = directory if isinstance(directory, list) else [directory]
        self.Modules = modules
        self.Operation = operation
        self.Pool = []
        # Find all NPZ files in the directory and subdirectories
        for dir in self.Directory:
            for root, _dirs, files in os.walk(dir):
                for file in files:
                    if file.lower().endswith(".npz"):
                        if max_files is None or len(self.Pool) < max_files:
                            self.Pool.append(os.path.join(root, file))
        self.Pool.sort()
        self.Filter()

    def GetName(self, file):
        return os.path.splitext(os.path.basename(file))[0]

    def LoadMotion(self, index):
        motion = Motion.LoadFromNPZ(self.Files[index], operation=self.Operation)
        if self.Modules is not None:
            motion.AddModules(self.Modules)
        return motion
        # return self.Instance(motion, index)

    def LoadInstance(self, index):
        return self.Instance(self.LoadMotion(index), index)

    def Filter(self, id=None):
        self.Files = []
        for item in self.Pool:
            if id is None or id in self.GetName(item):
                self.Files.append(item)

    def __len__(self):
        return len(self.Files)
