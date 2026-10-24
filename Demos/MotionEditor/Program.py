# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
from pathlib import Path

from ai4animation import (
    AI4Animation,
    ContactModule,
    Dataset,
    GuidanceModule,
    MirrorModule,
    MotionEditor,
    MotionModule,
    RootModule,
    Utility,
    Vector3,
)

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_/Cranberry")

Definitions = Utility.LoadModule(
    os.path.join(ASSETS_PATH, "Definitions.py"), "Definitions"
)


class Program:
    def Start(self):
        editor = AI4Animation.Scene.AddEntity("MotionEditor")

        editor.AddComponent(
            MotionEditor,
            Dataset(
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
                    lambda x: ContactModule(
                        x,
                        [
                            (Definitions.LeftAnkleName, 0.25, 0.02),
                            (Definitions.LeftBallName, 0.25, 0.01),
                            (Definitions.RightAnkleName, 0.25, 0.02),
                            (Definitions.RightBallName, 0.25, 0.01),
                        ],
                    ),
                    lambda x: GuidanceModule(x),
                    lambda x: MirrorModule(
                        x, Vector3.Axis.ZPositive, Vector3.Create(0, 0, 180)
                    ),
                ],
            ),
            os.path.join(ASSETS_PATH, "Model.glb"),
            Definitions.FULL_BODY_NAMES,
        )

        AI4Animation.Standalone.Camera.SetTarget(editor)

    def Update(self):
        pass


def main():
    AI4Animation(Program())


if __name__ == "__main__":
    main()
