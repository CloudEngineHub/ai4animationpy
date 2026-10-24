# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
from pathlib import Path

import torch
from ai4animation import (
    Actor,
    AI4Animation,
    CosineAnnealingOptimizer,
    DataSampler,
    Dataset,
    FeedTensor,
    Losses,
    MirrorModule,
    MotionEditor,
    MotionModule,
    Plotting,
    ReadTensor,
    RootModule,
    Rotation,
    RunningStatistics,
    Tensor,
    TimeSeries,
    Transform,
    Utility,
    Vector3,
)
from ai4animation.AI.Library.Blocks import LinearBlock

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")

Definitions = Utility.LoadModule(
    os.path.join(ASSETS_PATH, "Definitions.py"), "Definitions"
)

EPOCH_COUNT = 150
BATCH_SIZE = 32
FRAMERATE = 30
DRAW_INTERVAL = 500
BONES = Definitions.FULL_BODY_NAMES
FEATURE_DIM = 9 * len(BONES)
LOCAL_TRANSFORM_DIM = 9 * len(BONES)
HIDDEN_DIM = 512
LATENT_DIM = 256


class ForwardKinematics(torch.nn.Module):
    def __init__(self, skeleton):
        super().__init__()
        self.Skeleton = skeleton
        self.BoneCount = skeleton.Hierarchy.BoneCount

    def forward(self, values):
        local_parameters = values.reshape(*values.shape[:-1], self.BoneCount, 9)
        offsets = local_parameters[..., :3]
        rotation_vectors = local_parameters[..., 3:]
        z = torch.nn.functional.normalize(rotation_vectors[..., :3], dim=-1, eps=1e-6)
        y = rotation_vectors[..., 3:]
        y = y - torch.sum(y * z, dim=-1, keepdim=True) * z
        y = torch.nn.functional.normalize(y, dim=-1, eps=1e-6)
        x = torch.cross(y, z, dim=-1)
        local_rotations = torch.stack((x, y, z), dim=-1)
        local_transforms = Transform.TR(offsets, local_rotations)
        root = torch.eye(
            4, dtype=local_transforms.dtype, device=local_transforms.device
        )
        global_transforms = self.Skeleton.ComputeFK(local_transforms, root=root)
        feature_shape = global_transforms.shape[:-3] + (-1,)
        return torch.cat(
            (
                Transform.GetPosition(global_transforms).reshape(feature_shape),
                Transform.GetAxisZ(global_transforms).reshape(feature_shape),
                Transform.GetAxisY(global_transforms).reshape(feature_shape),
            ),
            dim=-1,
        )


class Model(torch.nn.Module):
    def __init__(self, skeleton):
        super().__init__()
        self.Statistics = RunningStatistics(FEATURE_DIM)
        self.Encoder = LinearBlock(FEATURE_DIM, LATENT_DIM, HIDDEN_DIM)
        self.Decoder = LinearBlock(LATENT_DIM, LOCAL_TRANSFORM_DIM, HIDDEN_DIM)
        self.ForwardKinematics = ForwardKinematics(skeleton)

    def reconstruct(self, features):
        normalized_features = self.Statistics.Normalize(features)
        embedding = self.Encoder(normalized_features)
        local_parameters = self.Decoder(embedding)
        reconstruction = self.ForwardKinematics(local_parameters)
        return reconstruction, embedding, local_parameters

    def forward(self, features):
        reconstruction, _, _ = self.reconstruct(features)
        return reconstruction

    def learn(self, features, update_statistics):
        if update_statistics:
            self.Statistics.Update(features)

        normalized_features = self.Statistics.Normalize(features)
        reconstruction, embedding, _ = self.reconstruct(features)
        normalized_reconstruction = self.Statistics.Normalize(reconstruction)
        loss = Losses.MSE(normalized_reconstruction, normalized_features)
        return {"Y": reconstruction, "Z": embedding}, {"MSE Loss": loss}


class Program:
    def Start(self):
        Utility.SetSeed(23456)

        self.Dataset = Dataset(
            os.path.join(ASSETS_PATH, "Motions"),
            [
                lambda x: RootModule(
                    x,
                    Definitions.HipName,
                    Definitions.LeftHipName,
                    Definitions.RightHipName,
                    Definitions.LeftShoulderName,
                    Definitions.RightShoulderName,
                    Definitions.NeckName,
                ),
                lambda x: MotionModule(x),
                lambda x: MirrorModule(
                    x, Vector3.Axis.ZPositive, Vector3.Create(0, 0, 180)
                ),
            ],
        )

        self.DataSampler = DataSampler(
            self.Dataset,
            framerate=FRAMERATE,
            batch_size=BATCH_SIZE,
            function=self.GetTrainingFeatures,
        )

        self.LossHistory = Plotting.LossHistory(
            "Loss History",
            horizon=self.DataSampler.BatchCount,
            drawInterval=DRAW_INTERVAL,
            yScale="log",
        )
        self.Trainer = None
        self.LocalOffsets = None
        self.LocalRotationVectors = None

    def Standalone(self):
        self.Editor = AI4Animation.Scene.AddEntity("Editor").AddComponent(
            MotionEditor,
            self.Dataset,
            os.path.join(ASSETS_PATH, "Model.glb"),
            BONES,
        )
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor, os.path.join(ASSETS_PATH, "Model.glb"), BONES
        )
        self.Actor.SkinnedMesh.SetColor(AI4Animation.Color.RED)
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

        self.Network = Tensor.ToDevice(Model(self.Actor.Skeleton))
        self.Optimizer = CosineAnnealingOptimizer(
            self.Network.parameters(),
            self.DataSampler.BatchSize,
            self.DataSampler.BatchCount,
        )
        self.Trainer = self.Training()

    def Update(self):
        if self.Trainer is None:
            return
        try:
            next(self.Trainer)
        except StopIteration:
            pass

    def Training(self):
        for epoch in range(1, EPOCH_COUNT + 1):
            print("Epoch", epoch)
            for batch in self.DataSampler.SampleBatchesWithinMotions(
                epoch, EPOCH_COUNT
            ):
                _, loss = self.Network.learn(batch, epoch == 1)
                self.Optimizer.Update(loss)
                self.LossHistory.Add(loss)
                yield
            self.LossHistory.Print()

    def GetTrainingFeatures(self, batch):
        motion, timestamps = batch
        mirrored = Tensor.RandomBool()

        inputs = FeedTensor("X", (len(timestamps), FEATURE_DIM))

        window = Tensor.RandomUniform(min=0.0, max=1.0)
        smoothing = TimeSeries(-window / 2, window / 2, 10)
        rootInv = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(
                timestamps, mirrored=mirrored, smoothing=smoothing
            )
        )

        transforms = Transform.TransformationFrom(
            motion.GetBoneTransformations(timestamps, BONES, mirrored=mirrored),
            rootInv.reshape(-1, 1, 4, 4),
        )
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        return inputs.GetTensor()

    def GetEditorFeatures(self):
        features = FeedTensor("X", FEATURE_DIM)
        root = self.Editor.Actor.Skeleton.Root
        transforms = Transform.TransformationTo(
            self.Editor.Actor.Skeleton.GetTransforms(BONES), root
        )
        features.Feed(Transform.GetPosition(transforms))
        features.Feed(Transform.GetAxisZ(transforms))
        features.Feed(Transform.GetAxisY(transforms))
        return features.GetTensor()

    def Draw(self):
        self.Network.eval()
        with torch.no_grad():
            xBatch = self.GetEditorFeatures()
            prediction, _, local_parameters = self.Network.reconstruct(xBatch)
            local_parameters = local_parameters.reshape(len(BONES), 9)
            self.LocalOffsets = Tensor.ToNumPy(local_parameters[..., :3])
            self.LocalRotationVectors = Tensor.ToNumPy(
                local_parameters[..., 3:].reshape(len(BONES), 2, 3)
            )

            output = ReadTensor("Y", Tensor.ToNumPy(prediction))
            self.Actor.Skeleton.Root = self.Editor.Actor.Skeleton.Root
            self.Actor.Skeleton.SetPositions(
                Vector3.PositionFrom(
                    output.ReadVector3(len(BONES)), self.Actor.Skeleton.Root
                )
            )
            self.Actor.Skeleton.SetRotations(
                Rotation.RotationFrom(
                    output.ReadRotation3D(len(BONES)), self.Actor.Skeleton.Root
                )
            )
            self.Actor.SyncToScene()
        self.Network.train()

    def DrawParameterPivots(
        self,
        bone_index,
        x,
        y,
        width,
        height,
        offset_limit,
        rotation_limit,
        colors,
    ):
        AI4Animation.Draw.Text(
            BONES[bone_index],
            x + width / 2.0,
            y + 0.12 * height,
            size=0.14 * height,
            color=AI4Animation.Color.WHITE,
            pivot=0.5,
        )

        gap = 0.002
        pivot_width = (width - 2.0 * gap) / 3.0
        pivot_height = 0.18 * height
        labels = (("OX", "OY", "OZ"), ("AX", "AY", "AZ"), ("BX", "BY", "BZ"))
        values = (
            self.LocalOffsets[bone_index],
            self.LocalRotationVectors[bone_index, 0],
            self.LocalRotationVectors[bone_index, 1],
        )
        limits = (
            (-offset_limit, offset_limit),
            (-rotation_limit, rotation_limit),
            (-rotation_limit, rotation_limit),
        )
        for group in range(3):
            pivot_y = y + (0.24 + 0.25 * group) * height
            for axis in range(3):
                value = max(
                    limits[group][0], min(limits[group][1], values[group][axis])
                )
                AI4Animation.GUI.HorizontalPivot(
                    x + axis * (pivot_width + gap),
                    pivot_y,
                    pivot_width,
                    pivot_height,
                    value,
                    label=labels[group][axis],
                    limits=limits[group],
                    thickness=0.08,
                    pivotColor=colors[axis],
                )

    def GUI(self):
        if self.LocalOffsets is None or self.LocalRotationVectors is None:
            return

        panel_x, panel_y = 0.79, 0.05
        panel_width, panel_height = 0.20, 0.90
        column_gap = 0.006
        rows = (len(BONES) + 1) // 2
        column_width = (panel_width - column_gap) / 2.0
        row_height = panel_height / rows
        offset_limit = max(1.0, float(abs(self.LocalOffsets).max()))
        rotation_limit = max(1.0, float(abs(self.LocalRotationVectors).max()))
        colors = (
            AI4Animation.Color.RED,
            AI4Animation.Color.GREEN,
            AI4Animation.Color.BLUE,
        )

        for bone_index in range(len(BONES)):
            column = bone_index // rows
            row = bone_index % rows
            self.DrawParameterPivots(
                bone_index,
                panel_x + column * (column_width + column_gap),
                panel_y + row * row_height,
                column_width,
                row_height,
                offset_limit,
                rotation_limit,
                colors,
            )


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "__main__":
    main()
