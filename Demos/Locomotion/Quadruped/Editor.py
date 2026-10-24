# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
from pathlib import Path

from ai4animation import (
    AI4Animation,
    ContactModule,
    Dataset,
    GuidanceModule,
    MirrorModule,
    Motion,
    MotionEditor,
    MotionModule,
    RootModule,
    Rotation,
    Transform,
    Utility,
    Vector3,
)

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Quadruped")

Definitions = Utility.LoadModule(
    os.path.join(ASSETS_PATH, "Definitions.py"), "Definitions"
)


class Program:
    def Start(self):
        editor = AI4Animation.Scene.AddEntity("MotionEditor")
        self.dataset = Dataset(
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
                    topology=RootModule.Topology.QUADRUPED,
                ),
                lambda x: MotionModule(x),
                lambda x: ContactModule(
                    x,
                    [
                        (Definitions.LeftHandSiteName, 1.0),
                        (Definitions.RightHandSiteName, 1.0),
                        (Definitions.LeftFootSiteName, 1.0),
                        (Definitions.RightFootSiteName, 1.0),
                    ],
                ),
                lambda x: GuidanceModule(x),
                lambda x: MirrorModule(
                    x,
                    Vector3.Axis.ZPositive,
                ),
            ],
            operation=lambda x: self.CorrectMotion(x),
        )

        editor.AddComponent(
            MotionEditor,
            self.dataset,
            Motion.LoadFromNPZ(
                os.path.join(ASSETS_PATH, "Motions", "D1_008_KAN01_001")
            ).Hierarchy,
            Definitions.FULL_BODY_NAMES,
        )

        AI4Animation.Standalone.Camera.SetTarget(editor)

    def Update(self):
        pass

    def CorrectMotion(self, motion):
        motion.Frames = Transform.GetMirror(motion.Frames, Vector3.Axis.XPositive)
        idx = motion.Hierarchy.GetBoneIndices(
            [
                Definitions.HeadName,
                Definitions.LeftShoulderName,
                Definitions.RightShoulderName,
            ]
        )
        motion.Frames[:, idx] = Transform.Multiply(
            motion.Frames[:, idx], Transform.R(Rotation.Euler(90, 0, 0))
        )


def main():
    AI4Animation(Program())


if __name__ == "__main__":
    main()
