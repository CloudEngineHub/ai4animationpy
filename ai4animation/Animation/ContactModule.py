# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Module for computing bone contact labels based on velocity thresholds."""

from ai4animation import Utility
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Animation.Module import Module
from ai4animation.Animation.Motion import Motion
from ai4animation.Animation.TimeSeries import TimeSeries
from ai4animation.Math import Tensor


class ContactModule(Module):
    def __init__(
        self, motion: Motion, configs, proportional=False
    ) -> None:  # Each config is a tuple of (boneName, velocityThreshold)
        super().__init__(motion)

        self.Configs = configs
        self.BoneNames = [config[0] for config in configs]
        self.BoneIndices = self.Motion.Hierarchy.GetBoneIndices(self.BoneNames)
        self.VelocityThresholds = [config[1] for config in configs]
        self.HeightThresholds = [config[2] for config in configs]
        self.Proportional = proportional

        # self.Grounded = self.IsMotionOnGround()
        # print(self.Motion.Name, self.Grounded)

        for config in configs:
            if len(config) != 3:
                print(
                    "ContactModule config length did not have expected tuple size of 3 for (boneName, velocityThreshold, heightThreshold)"
                )

    def Initialize(self):
        pass

    def GetName(self):
        return "Contact"

    def ComputeSeries(
        self,
        timestamp: float,
        mirrored: bool,
        timeseries: TimeSeries,
    ):
        timestamps = timeseries.SimulateTimestamps(timestamp)
        instance = self.Series(
            timeseries,
            self.BoneNames,
            self.GetContacts(timestamps, mirrored),
            self.GetGrounded(timestamps, mirrored),
        )
        return instance

    def GUI(self, editor):
        if Module.Visualize[ContactModule]:
            self.ComputeSeries(editor.Timestamp, editor.Mirror, editor.TimeSeries).GUI(
                0.3, 0.9, 0.4, 0.05
            )

    def Draw(self, editor):
        if Module.Visualize[ContactModule]:
            timestamps = editor.TimeSeries.SimulateTimestamps(editor.Timestamp)
            positions = self.Motion.GetBonePositions(
                timestamps, self.BoneIndices, editor.Mirror
            ).reshape(-1, 3)
            values = self.GetContacts(timestamps, editor.Mirror).reshape(-1, 1)
            ground = self.GetGrounded(timestamps, editor.Mirror).reshape(-1, 1)
            for i in range(values.shape[0]):
                if values[i]:
                    AI4Animation.Draw.Sphere(
                        positions[i],
                        size=0.04,
                        color=Utility.Opacity(AI4Animation.Color.GREEN, 0.5),
                    )
                else:
                    AI4Animation.Draw.Sphere(
                        positions[i],
                        size=0.02,
                        color=Utility.Opacity(AI4Animation.Color.BLACK, 0.25),
                    )
                if ground[i]:
                    AI4Animation.Draw.Sphere(
                        positions[i],
                        size=0.04,
                        color=Utility.Opacity(AI4Animation.Color.RED, 0.5),
                    )
            self.DrawGroundPlanes(editor)

    def DrawGroundPlanes(self, editor):
        positions = self.Motion.GetBonePositions(
            editor.Timestamp, self.BoneIndices, editor.Mirror
        )
        if self.Proportional:
            lengths = self.Motion.GetBoneLengths(
                timestamps=editor.Timestamp, mirrored=editor.Mirror
            )
            scales = Tensor.Sum(lengths, axis=-2, keepDim=False)
        else:
            scales = 1
        positions[..., 1] = self.HeightThresholds * scales
        AI4Animation.Draw.Plane(
            positions, 0.25, Utility.Opacity(AI4Animation.Color.BLACK, 0.5)
        )

    def GetContacts(self, timestamps, mirrored):
        velocities = self.Motion.GetBoneVelocities(
            timestamps, self.BoneIndices, mirrored
        )
        velocities = Tensor.Norm(velocities, keepDim=False)
        if self.Proportional:
            lengths = self.Motion.GetBoneLengths(
                timestamps=timestamps, mirrored=mirrored
            )
            scales = Tensor.Sum(lengths, axis=-2, keepDim=False)
        else:
            scales = 1
        return velocities < (self.VelocityThresholds * scales)

    def GetGrounded(self, timestamps, mirrored):
        positions = self.Motion.GetBonePositions(timestamps, self.BoneIndices, mirrored)
        heights = positions[..., 1]
        if self.Proportional:
            lengths = self.Motion.GetBoneLengths(
                timestamps=timestamps, mirrored=mirrored
            )
            scales = Tensor.Sum(lengths, axis=-2, keepDim=False)
        else:
            scales = 1
        return heights < (self.HeightThresholds * scales)

    def IsMotionOnGround(self, duration=1.0):
        grounded = self.GetGrounded(None, False)
        any_grounded = grounded.any(axis=-1)
        window_size = max(1, int(duration * self.Motion.Framerate))
        n = len(any_grounded)
        if n < window_size:
            return bool(any_grounded.any())
        cs = any_grounded.cumsum()
        if cs[window_size - 1] == 0:
            return False
        if n > window_size:
            window_sums = cs[window_size:] - cs[: n - window_size]
            if window_sums.min() == 0:
                return False
        return True

    class Series(TimeSeries):
        def __init__(self, timeSeries, names, values=None, ground=None):
            super().__init__(timeSeries.Start, timeSeries.End, timeSeries.SampleCount)
            self.Names = names
            self.Values = (
                Tensor.Zeros((self.SampleCount, len(self.Names)))
                if values is None
                else values
            )
            self.Ground = ground

        def GUI(self, x=0.3, y=0.94, width=0.4, height=0.05):
            if self.Values is not None:
                AI4Animation.GUI.BarPlot(
                    x,
                    y,
                    width,
                    height,
                    Tensor.SwapAxes(self.Values, 0, 1),
                    label="Contacts",
                    colors=AI4Animation.Color.WHITE,
                )
            if self.Ground is not None:
                AI4Animation.GUI.BarPlot(
                    x,
                    y - height,
                    width,
                    height,
                    Tensor.SwapAxes(self.Ground, 0, 1),
                    label="Grounded",
                    colors=AI4Animation.Color.RED,
                )

        def Draw(self, motion):
            contact_bones = [
                (contact_index, motion.NameToIndexMap[name])
                for contact_index, name in enumerate(self.Names)
                if name in motion.NameToIndexMap
            ]
            positions = motion.GetPositions()
            contact_color = Utility.Opacity(AI4Animation.Color.GREEN, 0.5)
            inactive_color = Utility.Opacity(AI4Animation.Color.BLACK, 0.25)
            ground_color = Utility.Opacity(AI4Animation.Color.RED, 0.5)

            for sample_index in range(self.SampleCount):
                for contact_index, bone_index in contact_bones:
                    position = positions[sample_index, bone_index]
                    if self.Values[sample_index, contact_index]:
                        AI4Animation.Draw.Sphere(
                            position,
                            size=0.04,
                            color=contact_color,
                        )
                    else:
                        AI4Animation.Draw.Sphere(
                            position,
                            size=0.02,
                            color=inactive_color,
                        )
                    if (
                        self.Ground is not None
                        and self.Ground[sample_index, contact_index]
                    ):
                        AI4Animation.Draw.Sphere(
                            position,
                            size=0.04,
                            color=ground_color,
                        )
