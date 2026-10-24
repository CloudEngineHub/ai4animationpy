# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
from pathlib import Path

import numpy as np
from ai4animation import Actor, AI4Animation, Rotation, Transform, Utility

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_/Cranberry")

Definitions = Utility.LoadModule(
    os.path.join(ASSETS_PATH, "Definitions.py"), "Definitions"
)


class Program:
    def Start(self):
        entity = AI4Animation.Scene.AddEntity("Actor")
        self.Actor = entity.AddComponent(
            Actor, os.path.join(ASSETS_PATH, "Model.glb"), Definitions.FULL_BODY_NAMES
        )

        self.Hierarchy = self.Actor.Skeleton.Hierarchy
        root_index = self.Hierarchy.ParentIndices.index(None)
        self.Root = self.Actor.Skeleton.GetTransforms(root_index).copy()
        self.BindLocalTransforms = self.Actor.Skeleton.ZeroTransforms.copy()
        self.JointAngles = np.zeros((self.Hierarchy.BoneCount, 3), dtype=np.float32)

    def Standalone(self):
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

        self.Canvas = AI4Animation.GUI.Canvas(
            "Forward Kinematics", 0.75, 0.03, 0.24, 0.94
        )
        self.Sliders = []
        self.JointRows = []
        top = 0.04
        row_height = 0.0325
        self.SliderHeight = 0.0275
        column_x = (0.41, 0.605, 0.80)
        slider_width = 0.185
        for joint_index, joint_name in enumerate(self.Hierarchy.BoneNames):
            row = []
            y = top + joint_index * row_height
            self.JointRows.append((joint_name, y))
            for axis_index, axis_name in enumerate(("X", "Y", "Z")):
                row.append(
                    AI4Animation.GUI.Slider(
                        column_x[axis_index],
                        y,
                        slider_width,
                        self.SliderHeight,
                        0.0,
                        -180.0,
                        180.0,
                        canvas=self.Canvas,
                        label=axis_name,
                    )
                )
            self.Sliders.append(row)

        self.ResetButton = AI4Animation.GUI.Button(
            "Reset",
            0.41,
            0.93,
            0.575,
            0.045,
            state=False,
            toggle=False,
            canvas=self.Canvas,
        )

    def Update(self):
        local_transforms = self.BindLocalTransforms.copy()
        delta_rotations = Rotation.Euler(self.JointAngles)
        local_rotations = Rotation.RotationFrom(delta_rotations, local_transforms)
        Transform.SetRotation(local_transforms, local_rotations)

        pose = self.Actor.Skeleton.ComputeFK(local_transforms, root=self.Root)
        self.Actor.Skeleton.SetTransforms(pose)
        self.Actor.SyncToScene()

    def Draw(self):
        pass

    def GUI(self):
        self.Canvas.GUI()
        for joint_name, y in self.JointRows:
            AI4Animation.Draw.Text(
                joint_name,
                0.01,
                y + 0.25 * self.SliderHeight,
                size=0.5 * self.Canvas.Rectangle.height * self.SliderHeight,
                color=AI4Animation.Color.WHITE,
                canvas=self.Canvas,
            )

        for joint_index, row in enumerate(self.Sliders):
            for axis_index, slider in enumerate(row):
                self.JointAngles[joint_index, axis_index] = slider.GUI()

        self.ResetButton.GUI()
        if self.ResetButton.IsPressed():
            self.JointAngles.fill(0.0)
            for row in self.Sliders:
                for slider in row:
                    slider.SetValue(0.0)


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "__main__":
    main()
