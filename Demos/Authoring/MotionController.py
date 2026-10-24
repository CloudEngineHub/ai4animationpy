# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_/Geno")

from ai4animation import (
    Actor,
    AI4Animation,
    FeedTensor,
    GuidanceModule,
    MotionModule,
    ReadTensor,
    RootModule,
    Rotation,
    Tensor,
    Time,
    TimeSeries,
    Transform,
    Utility,
    Vector3,
)

Definitions = Utility.LoadModule(
    os.path.join(ASSETS_PATH, "Definitions.py"), "Definitions"
)
from LegIK import LegIK
from Sequence import Sequence


class MotionController:
    def __init__(self):
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor, os.path.join(ASSETS_PATH, "Model.glb"), Definitions.FULL_BODY_NAMES
        )
        self.Model = torch.load(
            os.path.join(SCRIPT_DIR, "Models/Network.pt"), weights_only=False
        )
        self.Model.eval()
        self.PostProcessor = torch.load(
            os.path.join(SCRIPT_DIR, "Models/PostProcessor.pt"), weights_only=False
        )
        self.PostProcessor.eval()
        self.SequenceWindow = 0.5
        self.SequenceLength = 16
        self.SequenceFPS = 30
        self.MaxTimescale = 1.5
        self.TimescaleSensitivity = 5.0
        self.Timescale = 1.0
        self.SolverIterations = 1
        self.SolverAccuracy = 1e-3
        self.ContactPower = 3.0
        self.NetworkIterations = 3
        self.Synchronization = 0.0
        self.Timescale = 1.0
        self.TrajectoryCorrection = 0.25
        self.Timestamp = Time.TotalTime

        self.ControlSeries = TimeSeries(0.0, self.SequenceWindow, self.SequenceLength)
        self.SimulationObject = RootModule.Series(self.ControlSeries)
        self.RootControl = RootModule.Series(self.ControlSeries)
        self.GuidanceControl = GuidanceModule.Guidance(
            "Guidance",
            list(self.Actor.Skeleton.Hierarchy.BoneNames),
            self.Actor.Skeleton.GetPositions().copy(),
        )
        self.GuidanceTemplates = {}
        directory = os.path.join(SCRIPT_DIR, "Guidances")
        for path in sorted(os.listdir(directory)):
            with np.load(os.path.join(directory, path), allow_pickle=True) as data:
                id = Path(path).stem
                names = data["Names"]
                positions = data["Positions"]
                self.GuidanceTemplates[id] = GuidanceModule.Guidance(
                    id, names, positions
                )
                print("Added Guidance:", id)

        self.GuidanceNames = sorted(self.GuidanceTemplates.keys())
        self.GuidanceControl.Positions = self.GuidanceTemplates["Idle"].Positions.copy()

        self.ContactBones = [
            Definitions.LeftAnkleName,
            Definitions.LeftBallName,
            Definitions.RightAnkleName,
            Definitions.RightBallName,
        ]
        self.ContactIndices = self.Actor.Skeleton.Hierarchy.GetBoneIndices(
            self.ContactBones
        )

        self.LeftLegIK = LegIK(
            self.Actor.Skeleton.CreateIK(
                Definitions.LeftHipName,
                Definitions.LeftAnkleName,
            ),
            self.Actor.Skeleton.CreateIK(
                Definitions.LeftAnkleName,
                Definitions.LeftBallName,
            ),
        )

        self.RightLegIK: LegIK = LegIK(
            self.Actor.Skeleton.CreateIK(
                Definitions.RightHipName,
                Definitions.RightAnkleName,
            ),
            self.Actor.Skeleton.CreateIK(
                Definitions.RightAnkleName,
                Definitions.RightBallName,
            ),
        )
        self.Previous = None
        self.Sequence = None

    def Update(self, goal, control_strength, guidance_pose, dt, prediction_frequency):
        # Update control every frame
        self.Control(goal, control_strength, guidance_pose)

        # Predict future sequence every few frames
        if self.Timestamp == 0.0 or Time.TotalTime - self.Timestamp > (
            1.0 / prediction_frequency
        ):
            self.Timestamp = Time.TotalTime
            self.PredictSequence()

        # Animate motion every frame
        if self.Sequence is not None:
            self.Animate(dt, prediction_frequency)

    def Control(self, goal, control_strength, guidance_pose):
        self.ControlFromTarget(
            self.SimulationObject,
            Transform.Interpolate(
                self.Actor.Skeleton.Root,
                self.SimulationObject.Transforms[0],
                0.5,
            ),
            goal,
            control_strength,
        )
        speed = Vector3.Length(self.SimulationObject.GetVelocity(0))

        if speed < 0.1:
            self.GuidanceControl.Positions = self.GuidanceTemplates[
                "Idle"
            ].Positions.copy()
        else:
            self.GuidanceControl.Positions = guidance_pose.copy()

        # Correction
        if self.Sequence is not None:
            # Trajectory
            self.RootControl.Transforms = Transform.Interpolate(
                self.SimulationObject.Transforms,
                self.Sequence.Trajectory.Transforms,
                self.TrajectoryCorrection,
            )
            for i in range(self.RootControl.SampleCount):
                target = Transform.GetPosition(self.RootControl.Transforms)[i:]
                current = self.Actor.Skeleton.GetRootPosition().reshape(-1, 3)
                time = self.RootControl.Timestamps[i:].reshape(-1, 1)
                self.RootControl.Velocities[i] = Tensor.Sum(
                    target - current, axis=0, keepDim=False
                ) / Tensor.Sum(time, axis=0, keepDim=False)
            self.RootControl.Velocities = Vector3.Lerp(
                self.RootControl.Velocities,
                self.Sequence.Trajectory.Velocities,
                self.TrajectoryCorrection,
            )

    def PredictSequence(self):
        inputs = FeedTensor("X", self.Model.input_dim())

        root = self.Actor.Skeleton.Root

        transforms = Transform.TransformationTo(
            self.Actor.Skeleton.GetTransforms(), root
        )
        velocities = Vector3.DirectionTo(self.Actor.Skeleton.GetVelocities(), root)
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)

        futureRootTransforms = Transform.TransformationTo(
            self.RootControl.Transforms, root
        )
        futureRootVelocities = Vector3.DirectionTo(self.RootControl.Velocities, root)

        inputs.FeedVector3(
            Transform.GetPosition(futureRootTransforms), x=True, y=False, z=True
        )
        inputs.FeedVector3(
            Transform.GetAxisZ(futureRootTransforms), x=True, y=False, z=True
        )
        inputs.FeedVector3(futureRootVelocities, x=True, y=False, z=True)

        inputs.Feed(self.GuidanceControl.Positions)

        outputs = self.Model(
            inputs.GetTensor().reshape(1, -1), iterations=self.NetworkIterations
        )
        outputs = outputs.reshape(self.SequenceLength, -1)
        outputs = ReadTensor("Y", Tensor.ToNumPy(outputs))

        # Generate Sequence
        futureRootVectors = outputs.ReadVector3()
        futureRootDelta = Tensor.ZerosLike(futureRootVectors)
        for i in range(1, self.SequenceLength):
            futureRootDelta[i] = futureRootDelta[i - 1] + futureRootVectors[i]
        futureRootTransforms = Transform.TransformationFrom(
            Transform.DeltaXZ(futureRootDelta), root
        )
        futureRootVelocities = Tensor.ZerosLike(futureRootVectors)
        futureRootVelocities[..., [0, 2]] = (
            futureRootVectors[..., [0, 2]] * self.SequenceFPS
        )
        futureRootVelocities = Vector3.DirectionFrom(
            futureRootVelocities, futureRootTransforms
        )

        futureMotionTransforms = Transform.TransformationFrom(
            Transform.TR(
                outputs.ReadVector3(self.Actor.Skeleton.Hierarchy.BoneCount),
                outputs.ReadRotation3D(self.Actor.Skeleton.Hierarchy.BoneCount),
            ),
            futureRootTransforms.reshape(self.SequenceLength, 1, 4, 4),
        )
        futureMotionVelocities = Vector3.DirectionFrom(
            outputs.ReadVector3(self.Actor.Skeleton.Hierarchy.BoneCount),
            futureRootTransforms.reshape(self.SequenceLength, 1, 4, 4),
        )

        self.Previous = self.Sequence
        self.Sequence = Sequence()
        self.Previous = self.Sequence if self.Previous is None else self.Previous
        self.Sequence.Timestamps = Tensor.LinSpace(
            0.0, self.SequenceWindow, self.SequenceLength
        )
        self.Sequence.Trajectory = RootModule.Series(
            self.ControlSeries, futureRootTransforms, futureRootVelocities
        )
        self.Sequence.Motion = MotionModule.Series(
            self.ControlSeries,
            list(self.Actor.Skeleton.Hierarchy.BoneNames),
            futureMotionTransforms,
            futureMotionVelocities,
        )

        # Predict Contacts
        inputs = FeedTensor("X", self.PostProcessor.input_dim())

        currentTransforms = self.Actor.Skeleton.GetTransforms(self.ContactIndices)
        currentVelocities = self.Actor.Skeleton.GetVelocities(self.ContactIndices)
        targetTransforms = self.Sequence.Motion.GetTransforms(self.ContactBones)[
            1:, :, :
        ]
        targetVelocities = self.Sequence.Motion.GetVelocities(self.ContactBones)[
            1:, :, :
        ]
        delta_distances = Vector3.Distance(
            Transform.GetPosition(currentTransforms),
            Transform.GetPosition(targetTransforms),
        )
        delta_angles = Rotation.Angle(
            Transform.GetRotation(currentTransforms),
            Transform.GetRotation(targetTransforms),
        )
        delta_velocities = Vector3.Distance(currentVelocities, targetVelocities)

        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)
        inputs.Feed(delta_distances)
        inputs.Feed(delta_angles)
        inputs.Feed(delta_velocities)

        contacts = Tensor.ToNumPy(
            self.PostProcessor(inputs.GetTensor()).reshape(
                self.SequenceLength, len(self.ContactBones)
            )
        )
        self.Sequence.Contacts = Tensor.Pow(
            Tensor.Clamp(contacts, 0, 1), self.ContactPower
        )

    def Animate(self, dt, prediction_frequency):
        requiredSpeed = (
            Vector3.Distance(
                self.Actor.Skeleton.GetRootPosition(),
                self.SimulationObject.GetPosition(0),
            )
            + self.SimulationObject.GetLength()
        ) / self.SequenceWindow
        predictedSpeed = self.Sequence.GetLength() / self.SequenceWindow
        if requiredSpeed > 0.1 and predictedSpeed > 0.1:
            ts = requiredSpeed / predictedSpeed
            sync = 1.0
        else:
            ts = 1.0
            sync = 0.0
        self.Timescale = Tensor.InterpolateDt(
            self.Timescale, ts, dt, self.TimescaleSensitivity
        )
        self.Timescale = Tensor.Clamp(self.Timescale, 1.0, self.MaxTimescale)
        self.Synchronization = Tensor.InterpolateDt(
            self.Synchronization, sync, dt, self.TimescaleSensitivity
        )

        sdt = dt * self.Timescale

        blend = (Time.TotalTime - self.Timestamp) * prediction_frequency
        root = Transform.Interpolate(
            self.Previous.SampleRoot(sdt), self.Sequence.SampleRoot(sdt), blend
        )
        positions = Vector3.Lerp(
            self.Previous.SamplePositions(sdt),
            self.Sequence.SamplePositions(sdt),
            blend,
        )
        rotations = Rotation.Interpolate(
            self.Previous.SampleRotations(sdt),
            self.Sequence.SampleRotations(sdt),
            blend,
        )
        velocities = Vector3.Lerp(
            self.Previous.SampleVelocities(sdt),
            self.Sequence.SampleVelocities(sdt),
            blend,
        )
        contacts = Tensor.Interpolate(
            self.Previous.SampleContacts(sdt), self.Sequence.SampleContacts(sdt), blend
        )

        self.Actor.Skeleton.Root = Transform.Interpolate(
            root, self.Actor.Skeleton.Root, self.Sequence.GetRootLock()
        )
        self.Actor.Skeleton.SetTransforms(
            Transform.TR(
                Vector3.Lerp(
                    self.Actor.Skeleton.GetPositions() + velocities * sdt,
                    positions,
                    0.5,
                ),
                rotations,
            )
        )
        self.Actor.Skeleton.SetVelocities(velocities)

        self.Actor.Skeleton.RestoreBoneLengths()
        self.Actor.Skeleton.RestoreBoneAlignments()

        self.LeftLegIK.Solve(
            ankleContact=contacts[0],
            ballContact=contacts[1],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
            poleTarget=Vector3.PositionFrom(
                Vector3.Create(0.0, 0.0, 1.0),
                self.Actor.Skeleton.GetTransforms(
                    self.Actor.Skeleton.Hierarchy.GetBoneIndices(
                        Definitions.LeftKneeName
                    )
                ),
            ),
            poleWeight=1.0,
        )
        self.RightLegIK.Solve(
            ankleContact=contacts[2],
            ballContact=contacts[3],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
            poleTarget=Vector3.PositionFrom(
                Vector3.Create(0.0, 0.0, 1.0),
                self.Actor.Skeleton.GetTransforms(
                    self.Actor.Skeleton.Hierarchy.GetBoneIndices(
                        Definitions.RightKneeName
                    )
                ),
            ),
            poleWeight=1.0,
        )

        self.Actor.SyncToScene()

        self.Previous.Timestamps -= sdt
        self.Sequence.Timestamps -= sdt

    def ControlFromTarget(self, trajectory, start, goal, strength=1.0):
        start_position = Transform.GetPosition(start).copy()
        goal_position = Transform.GetPosition(goal).copy()
        start_position[..., 1] = 0.0
        goal_position[..., 1] = 0.0

        delta = goal_position - start_position
        start_direction = Transform.GetAxisZ(start).copy()
        start_direction[..., 1] = 0.0
        if Vector3.Length(delta) != 0.0:
            goal_direction = delta
        else:
            goal_direction = Transform.GetAxisZ(goal).copy()
            goal_direction[..., 1] = 0.0
        if Vector3.Length(goal_direction) == 0.0:
            goal_direction = start_direction
        if Vector3.Length(start_direction) == 0.0:
            start_direction = goal_direction

        weight = Tensor.Unsqueeze(Tensor.LinSpace(0.0, 1.0, trajectory.SampleCount), -1)
        positions = Vector3.Lerp(start_position, goal_position, weight)
        directions = Vector3.Normalize(
            Vector3.Lerp(start_direction, goal_direction, weight)
        )
        trajectory.Transforms = Transform.TR(positions, Rotation.LookPlanar(directions))
        trajectory.Velocities = Tensor.Repeat(
            (strength * delta / max(trajectory.Window, Tensor.EPS)).reshape(1, 3),
            trajectory.SampleCount,
            0,
        )
