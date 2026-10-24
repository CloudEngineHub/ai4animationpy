# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Module for extracting per-bone motion features (positions, rotations, velocities)."""

import numpy as np
from ai4animation import Utility
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Animation.Module import Module
from ai4animation.Animation.Motion import Motion
from ai4animation.Animation.RootModule import RootModule
from ai4animation.Animation.TimeSeries import TimeSeries
from ai4animation.Math import Quaternion, Rotation, Tensor, Transform, Vector3

MAX_SMOOTHING = 2.0
MIN_POWER = 0.0
MAX_POWER = 1.0
MIN_SHIFT = 0.0
MAX_SHIFT = 0.5
MAX_NOISE = 0.025


class MotionModule(Module):
    def __init__(self, motion: Motion) -> None:
        super().__init__(motion)

    def Initialize(self):
        pass

    def GetName(self):
        return "Motion"

    def ComputeSeries(
        self,
        timestamp: float,
        mirrored: bool,
        names: list[str],
        timeseries: TimeSeries,
        smoothing: TimeSeries = None,
        power: float = 1.0,
        noise: float = 0.0,
        connect: bool = False,
    ):
        timestamps = timeseries.SimulateTimestamps(timestamp)
        transforms = self.GetTransforms(
            timestamps,
            mirrored,
            names,
            smoothing,
            power,
            noise,
            timestamp if connect else None,
        )
        # transforms = Transform.TR(
        #     self.GetPositions(
        #         timestamps,
        #         mirrored,
        #         names,
        #         smoothing,
        #         power,
        #         noise,
        #         timestamp if connect else None,
        #     ),
        #     self.GetRotations(
        #         timestamps,
        #         mirrored,
        #         names,
        #         smoothing,
        #         power,
        #         noise,
        #         timestamp if connect else None,
        #     ),
        # )
        velocities = self.GetVelocities(
            timestamps,
            mirrored,
            names,
            smoothing,
            power,
            noise,
            timestamp if connect else None,
        )
        instance = self.Series(timeseries, names, transforms, velocities)
        return instance

    def GetTransforms(
        self,
        timestamps,
        mirrored,
        names,
        smoothing: TimeSeries = None,
        power: float = 1.0,
        noise: float = 0.0,
        connect: float = None,
    ):
        if smoothing is not None and smoothing.Window > 0.0:
            curves = Transform.Normalize(
                self.SmoothCurves(
                    self.Motion.GetBoneTransformations,
                    timestamps,
                    mirrored,
                    names,
                    smoothing,
                    power,
                )
            )
        else:
            curves = self.Motion.GetBoneTransformations(timestamps, names, mirrored)
        if noise > 0.0:
            # Position
            curves[..., :3, 3] += noise * np.random.randn(*curves[..., :3, 3].shape)
            # Rotation
            angles = 180 * noise * Tensor.RandomUniform((*curves.shape[:-2], 3))
            delta = Rotation.Euler(angles)
            curves[..., :3, :3] = Rotation.RotationFrom(delta, curves[..., :3, :3])
        if connect is not None:
            curves = self.ConnectTransforms(
                curves=curves,
                source=curves[..., 0, :, :, :],
                target=self.GetTransforms(connect, mirrored, names),
                axis=-4,
            )
        return curves

    def GetPositions(
        self,
        timestamps,
        mirrored,
        names,
        smoothing: TimeSeries = None,
        power: float = 1.0,
        noise: float = 0.0,
        connect: float = None,
    ):
        if smoothing is not None and smoothing.Window > 0.0:
            curves = self.SmoothCurves(
                self.Motion.GetBonePositions,
                timestamps,
                mirrored,
                names,
                smoothing,
                power,
            )
        else:
            curves = self.Motion.GetBonePositions(timestamps, names, mirrored)
        if noise > 0.0:
            curves += noise * np.random.randn(*curves.shape)
        if connect is not None:
            curves = self.ConnectPositions(
                curves=curves,
                source=curves[..., 0, :, :],
                target=self.GetPositions(connect, mirrored, names),
                axis=-3,
            )
        return curves

    def GetRotations(
        self,
        timestamps,
        mirrored,
        names,
        smoothing: TimeSeries = None,
        power: float = 1.0,
        noise: float = 0.0,
        connect: float = None,
    ):
        if smoothing is not None and smoothing.Window > 0.0:
            curves = Rotation.Normalize(
                self.SmoothCurves(
                    self.Motion.GetBoneRotations,
                    timestamps,
                    mirrored,
                    names,
                    smoothing,
                    power,
                )
            )
        else:
            curves = self.Motion.GetBoneRotations(timestamps, names, mirrored)
        if noise > 0.0:
            angles = 180 * noise * Tensor.RandomUniform((*curves.shape[:-2], 3))
            delta = Rotation.Euler(angles)
            curves = Rotation.RotationFrom(delta, curves)
        if connect is not None:
            curves = self.ConnectRotations(
                curves=curves,
                source=curves[..., 0, :, :, :],
                target=Tensor.Squeeze(self.GetRotations(connect, mirrored, names), -4),
                axis=-4,
            )
        return curves

    def GetVelocities(
        self,
        timestamps,
        mirrored,
        names,
        smoothing: TimeSeries = None,
        power: float = 1.0,
        noise: float = 0.0,
        connect: float = None,
    ):
        dt = self.Motion.DeltaTime
        a = self.GetPositions(
            timestamps - dt,
            mirrored,
            names,
            smoothing,
            power,
            noise,
            # connect,
            False,
        )
        b = self.GetPositions(
            timestamps,
            mirrored,
            names,
            smoothing,
            power,
            noise,
            # connect,
            False,
        )
        return (b - a) / dt

    @staticmethod
    def ConnectTransforms(curves, source, target, axis, weight=1.0):
        T = MotionModule.ConnectPositions(
            Transform.GetPosition(curves),
            Transform.GetPosition(source),
            Transform.GetPosition(target),
            axis + 1,
            weight,
        )
        R = MotionModule.ConnectRotations(
            Transform.GetRotation(curves),
            Transform.GetRotation(source),
            Transform.GetRotation(target),
            axis,
            weight,
        )
        return Transform.TR(T, R)

    @staticmethod
    def ConnectPositions(curves, source, target, axis, weight=1.0):
        source = Tensor.Unsqueeze(source, axis)
        target = Tensor.Unsqueeze(target, axis)
        weights = Tensor.LinSpace(weight, 0.0, curves.shape[axis])
        T = target - source
        shape = [1] * T.ndim
        shape[axis] = -1
        weights = weights.reshape(shape)
        T = Vector3.Lerp(Vector3.Zero(), T, weights)
        T = curves + T
        return T

    @staticmethod
    def ConnectRotations(curves, source, target, axis, weight=1.0):
        source = Tensor.Unsqueeze(source, axis)
        target = Tensor.Unsqueeze(target, axis)
        weights = Tensor.LinSpace(weight, 0.0, curves.shape[axis])
        R = Rotation.RotationTo(target, source)
        shape = [1] * R.ndim
        shape[axis] = -1
        weights = weights.reshape(shape)
        R = Rotation.Interpolate(Rotation.Identity(), R, weights)
        R = Rotation.RotationFrom(R, curves)
        return R

    def SmoothCurves(self, fn, timestamps, mirrored, names, smoothing, power):
        axis = len(timestamps.shape)
        timestamps = Tensor.Unsqueeze(timestamps, -1)
        timestamps = timestamps + smoothing.Timestamps
        values = fn(timestamps, names, mirrored)
        values = Tensor.Gaussian(
            values,
            power=power,
            axis=axis,
            keepDim=False,
        )
        return values

    def Standalone(self):
        x = 0.325
        y = 0.165
        w = 0.24
        h = 0.04

        y += 0
        self.Slider_SmoothWindow = AI4Animation.GUI.Slider(
            x, y, w / 2, h, 0.0, 0.0, MAX_SMOOTHING, label="Smooth Window"
        )
        self.Slider_SmoothPower = AI4Animation.GUI.Slider(
            x + w / 2, y, w / 2, h, 1.0, MIN_POWER, MAX_POWER, label="Smooth Power"
        )
        self.Button_SmoothRandomize = AI4Animation.GUI.Button(
            "Random", x + w, y, 0.1, h, False, True
        )

        y += h
        self.Slider_MinShift = AI4Animation.GUI.Slider(
            x,
            y,
            w / 2,
            h,
            MIN_SHIFT,
            MIN_SHIFT,
            MAX_SHIFT,
            label="Min Shift",
        )
        self.Slider_MaxShift = AI4Animation.GUI.Slider(
            x + w / 2,
            y,
            w / 2,
            h,
            MAX_SHIFT,
            MIN_SHIFT,
            MAX_SHIFT,
            label="Max Shift",
        )
        self.Button_ShiftRandomize = AI4Animation.GUI.Button(
            "Random", x + w, y, 0.1, h, False, True
        )

        y += h
        self.Slider_NoiseAmount = AI4Animation.GUI.Slider(
            x, y, w, h, 0.0, 0.0, MAX_NOISE, label="Noise Amount"
        )
        self.Button_NoiseRandomize = AI4Animation.GUI.Button(
            "Random", x + w, y, 0.1, h, False, True
        )

        y += h
        self.Button_ConnectTransforms = AI4Animation.GUI.Button(
            "Connect Transforms", x, y, w, h, state=False
        )

    def GUI(self, editor):
        if Module.Visualize[MotionModule]:
            self.Slider_SmoothWindow.GUI()
            self.Slider_SmoothPower.GUI()
            self.Button_SmoothRandomize.GUI()

            self.Slider_MinShift.GUI()
            self.Slider_MaxShift.GUI()
            self.Button_ShiftRandomize.GUI()

            self.Slider_NoiseAmount.GUI()
            self.Button_NoiseRandomize.GUI()
            self.Button_ConnectTransforms.GUI()

    def Draw(self, editor):
        if Module.Visualize[MotionModule]:
            if self.Button_SmoothRandomize.Active:
                self.Slider_SmoothWindow.SetValue(
                    Tensor.RandomUniform(min=0.0, max=MAX_SMOOTHING)
                )
                # self.Slider_SmoothPower.SetValue(
                #     Tensor.RandomUniform(min=MIN_POWER, max=MAX_POWER)
                # )
            if self.Button_ShiftRandomize.Active:
                min = Tensor.RandomUniform(min=0.0, max=MAX_SHIFT)
                max = Tensor.RandomUniform(min=min, max=MAX_SHIFT)
                self.Slider_MinShift.SetValue(min)
                self.Slider_MaxShift.SetValue(max)
            if self.Button_NoiseRandomize.Active:
                self.Slider_NoiseAmount.SetValue(
                    Tensor.RandomUniform(min=0.0, max=MAX_NOISE)
                )

            window = self.Slider_SmoothWindow.GetValue()
            power = self.Slider_SmoothPower.GetValue()
            # window = np.power(window / MAX_SMOOTHING, 5.0) * MAX_SMOOTHING
            smoothing = (
                TimeSeries(
                    -window / 2,
                    window / 2,
                    editor.TimeSeries.SampleCount,
                )
                if window > 0.0
                else None
            )
            min_shift = self.Slider_MinShift.GetValue()
            max_shift = self.Slider_MaxShift.GetValue()
            noise = self.Slider_NoiseAmount.GetValue()
            series = self.ComputeSeries(
                editor.Timestamp,
                editor.Mirror,
                list(editor.Actor.Skeleton.Hierarchy.BoneNames),
                TimeSeries(min_shift, max_shift, 16),
                smoothing,
                power,
                noise,
                self.Button_ConnectTransforms.Active,
            )
            series.Draw()

    class Series(TimeSeries):
        def __init__(self, timeSeries, names, transforms=None, velocities=None):
            super().__init__(timeSeries.Start, timeSeries.End, timeSeries.SampleCount)
            self.Names = names
            self.NameToIndexMap = {}
            for i in range(len(names)):
                self.NameToIndexMap[names[i]] = i

            self.Transforms = (
                Transform.Identity((self.SampleCount, self.TrajectoryCount))
                if transforms is None
                else transforms
            )
            self.Velocities = (
                Vector3.Zero((self.SampleCount, self.TrajectoryCount))
                if velocities is None
                else velocities
            )

        @property
        def TrajectoryCount(self) -> int:
            return len(self.Names)

        def GetTransforms(self, bone_names=None, start=None, end=None):
            start = 0 if start is None else start
            end = self.SampleCount if end is None else end

            if bone_names == None:
                return self.Transforms[start:end]
            else:
                bone_indices = [self.NameToIndexMap[name] for name in bone_names]
                return self.Transforms[start:end, bone_indices, :, :]

        def GetPositions(self, bone_names=None, start=None, end=None):
            return Transform.GetPosition(self.GetTransforms(bone_names, start, end))

        def GetRotations(self, bone_names=None, start=None, end=None):
            return Transform.GetRotation(self.GetTransforms(bone_names, start, end))

        def GetVelocities(self, bone_names=None, start=None, end=None):
            start = 0 if start is None else start
            end = self.SampleCount if end is None else end

            if bone_names == None:
                return self.Velocities[start:end]
            else:
                bone_indices = [self.NameToIndexMap[name] for name in bone_names]
                return self.Velocities[start:end, bone_indices, :]

        def ClampDistance(self, pivot, distance):
            for index in range(self.SampleCount):
                for bone in range(self.TrajectoryCount):
                    offset = Transform.GetPosition(self.Transforms[index, bone]) - pivot
                    horizontal = Vector3.Create(offset[0], 0, offset[2])
                    if Vector3.Length(horizontal) > distance:
                        horizontal = distance * Vector3.Normalize(horizontal)
                        self.Transforms[index, bone, :3, 3] = pivot + Vector3.Create(
                            horizontal[0], offset[1], horizontal[2]
                        )

        def GetLengths(self):
            prev = Transform.GetPosition(self.Transforms)[..., :-1, :, :]
            next = Transform.GetPosition(self.Transforms)[..., 1:, :, :]
            values = Tensor.Sum(Tensor.Norm(next - prev, axis=0, keepDim=False))
            return values

        def Draw(
            self,
            start=None,
            end=None,
            thickness=1.0,
            drawConnections=True,
            drawPositions=True,
            drawRotations=False,
            drawVelocities=False,
            positionColor=None,
            rotationColor=None,
            velocityColor=None,
        ):
            start = 0 if start is None else start
            end = self.SampleCount if end is None else end
            _transforms_ = self.Transforms.reshape(-1, *self.Transforms.shape[-4:])
            _velocities_ = self.Velocities.reshape(-1, *self.Velocities.shape[-3:])
            for b in range(_transforms_.shape[0]):
                for idx, name in enumerate(self.Names):
                    transforms = _transforms_[b, start:end, idx, :, :]
                    velocities = _velocities_[b, start:end, idx, :]
                    positions = Transform.GetPosition(transforms)
                    rotations = Transform.GetRotation(transforms)
                    pColor = (
                        AI4Animation.Color.BLACK
                        if positionColor is None
                        else positionColor
                    )
                    vColor = Utility.Opacity(
                        (
                            AI4Animation.Color.GREEN
                            if velocityColor is None
                            else velocityColor
                        ),
                        0.5,
                    )
                    if drawConnections:
                        AI4Animation.Draw.LineStrip(
                            positions,
                            color=pColor,
                        )
                    if drawPositions:
                        AI4Animation.Draw.Sphere(
                            positions,
                            0.02 * thickness,
                            color=pColor,
                        )
                    if drawRotations:
                        AI4Animation.Draw.Transform(
                            transforms,
                            0.2 * thickness,
                            0.4 * thickness,
                        )
                    if drawVelocities:
                        AI4Animation.Draw.Vector(
                            positions,
                            velocities,
                            0.005 * thickness,
                            color=vColor,
                        )

        def DrawLegacy(
            self,
            start=None,
            end=None,
            thickness=1.0,
            drawConnections=True,
            drawPositions=True,
            drawRotations=False,
            drawVelocities=True,
            drawSkeleton=False,
            fade=False,
            positionColor=None,
            rotationColor=None,
            velocityColor=None,
            actor=None,
        ):
            start = 0 if start is None else start
            end = self.SampleCount if end is None else end

            # Per-sample view (history / anticipation): draw each frame's bones as
            # spheres, fading older/further samples, optionally with skeleton lines.
            if fade or drawSkeleton:
                count = max(end - start, 1)
                pColor = (
                    AI4Animation.Color.BLACK if positionColor is None else positionColor
                )
                for offset, sample in enumerate(range(start, end)):
                    alpha = (offset + 1) / count if fade else 1.0
                    color = Utility.Opacity(pColor, alpha)
                    positions = Transform.GetPosition(self.Transforms[sample])
                    if drawPositions:
                        AI4Animation.Draw.Sphere(
                            positions, 0.015 * thickness, color=color
                        )
                    if drawSkeleton and actor is not None:
                        AI4Animation.Draw.Skeleton(
                            None,
                            positions,
                            actor,
                            bones=self.Names,
                            color=color,
                        )
                return

            if actor is None:
                for i, _ in enumerate(self.Names):
                    transforms = self.Transforms[..., start:end, i, :, :]
                    positions = Transform.GetPosition(transforms)
                    rotations = Transform.GetRotation(transforms)
                    velocities = self.Velocities[..., start:end, i, :]
                    pColor = (
                        AI4Animation.Color.BLACK
                        if positionColor is None
                        else positionColor
                    )
                    vColor = Utility.Opacity(
                        (
                            AI4Animation.Color.GREEN
                            if velocityColor is None
                            else velocityColor
                        ),
                        0.5,
                    )
                    if drawConnections:
                        AI4Animation.Draw.LineStrip(positions, color=pColor)
                    if drawPositions:
                        AI4Animation.Draw.Sphere(
                            positions, 0.02 * thickness, color=pColor
                        )
                    if drawRotations:
                        AI4Animation.Draw.Transform(
                            transforms, 0.2 * thickness, 0.4 * thickness
                        )
                    if drawVelocities:
                        AI4Animation.Draw.Vector(
                            positions, velocities, 0.005 * thickness, color=vColor
                        )
            else:
                for i in range(start, end, 1):
                    AI4Animation.Draw.Skeleton(
                        None,
                        Transform.GetPosition(self.Transforms[i]),
                        actor,
                        bones=self.Names,
                    )
