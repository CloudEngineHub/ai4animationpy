# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
from pathlib import Path

from ai4animation import Actor, AI4Animation, Utility

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_/Cranberry")

Definitions = Utility.LoadModule(
    os.path.join(ASSETS_PATH, "Definitions.py"), "Definitions"
)


class Program:
    def Start(self):
        actor = AI4Animation.Scene.AddEntity("Actor")
        self.Actor = actor.AddComponent(
            Actor, os.path.join(ASSETS_PATH, "Model.glb"), Definitions.FULL_BODY_NAMES
        )

        self.IK = self.Actor.Skeleton.CreateIK(
            Definitions.LeftShoulderName, Definitions.LeftWristName
        )

        self.Target = AI4Animation.Scene.AddEntity("Target")
        wrist_index = self.Actor.Skeleton.Hierarchy.GetBoneIndices(
            Definitions.LeftWristName
        )
        self.Target.SetPosition(self.Actor.Skeleton.GetPositions(wrist_index))

        self.Pose = self.Actor.Skeleton.GetTransforms()

    def Standalone(self):
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

    def Update(self):
        self.Actor.Skeleton.SetTransforms(self.Pose)
        self.IK.Solve(
            self.Target.GetPosition(),
            self.Target.GetRotation(),
            max_iterations=10,
            threshold=0.001,
        )
        self.Actor.SyncToScene(self.IK.BoneIndices)

    def Draw(self):
        pass

    def GUI(self):
        self.Target.DrawHandle()
        pass


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "__main__":
    main()
