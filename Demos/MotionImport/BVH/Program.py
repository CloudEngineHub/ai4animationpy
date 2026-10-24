# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import sys
from pathlib import Path

from ai4animation import Actor, AI4Animation, Motion, Time, Utility, Vector3

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Geno")

Definitions = Utility.LoadModule(
    os.path.join(ASSETS_PATH, "Definitions.py"), "Definitions"
)

sys.path.append(Path(__file__).parent)

MOTION_FILE = "WalkingStickLeft_BR"


class Program:
    def __init__(self, filename):
        self.Filename = filename

    def Start(self):
        bvh_path = os.path.join(str(SCRIPT_DIR), self.Filename + ".bvh")
        # fbx_path = os.path.join(str(SCRIPT_DIR), self.Filename + ".fbx")
        # npz_path = os.path.join(str(SCRIPT_DIR), self.Filename + ".npz")
        model_path = os.path.join(ASSETS_PATH, "Model.glb")

        self.BVHMotion = Motion.LoadFromBVH(bvh_path, scale=0.01)
        # self.FBXMotion = Motion.LoadFromFBX(fbx_path)
        # self.NPZMotion = Motion.LoadFromNPZ(npz_path)
        self.BVHMotion.Debug()
        # self.FBXMotion.Debug()
        # self.NPZMotion.Debug()

        self.Mirror = False

        bvh_entity = AI4Animation.Scene.AddEntity("BVH Actor")
        self.BVHActor = bvh_entity.AddComponent(
            Actor, model_path, Definitions.FULL_BODY_NAMES
        )

        # fbx_entity = AI4Animation.Scene.AddEntity("FBX Actor")
        # self.FBXActor = fbx_entity.AddComponent(
        #     Actor, model_path, Definitions.FULL_BODY_NAMES
        # )

        # npz_entity = AI4Animation.Scene.AddEntity("NPZ Actor")
        # self.NPZActor = npz_entity.AddComponent(
        #     Actor, model_path, Definitions.FULL_BODY_NAMES
        # )

        # self.FBXActor.SkinnedMesh.SetColor(AI4Animation.Color.RED)
        # self.NPZActor.SkinnedMesh.SetColor(AI4Animation.Color.BLUE)

    def Standalone(self):
        AI4Animation.Standalone.Camera.SetTarget(self.BVHActor.Entity)

    def Update(self):
        bvh_timestamp = Time.TotalTime % self.BVHMotion.TotalTime
        self.BVHActor.Skeleton.SetTransforms(
            self.BVHMotion.GetBoneTransformations(
                timestamps=bvh_timestamp,
                bone_names_or_indices=list(self.BVHActor.Skeleton.Hierarchy.BoneNames),
                mirrored=self.Mirror,
            )
        )
        self.BVHActor.SyncToScene()

        # fbx_timestamp = Time.TotalTime % self.FBXMotion.TotalTime
        # self.FBXActor.Skeleton.SetTransforms(
        #     Transform.TransformationFrom(
        #         self.FBXMotion.GetBoneTransformations(
        #             timestamps=fbx_timestamp,
        #             bone_names_or_indices=list(self.FBXActor.Skeleton.Hierarchy.BoneNames),
        #             mirrored=self.Mirror,
        #         ),
        #         Transform.T(Vector3.Create(1, 0, 0))
        #     )
        # )
        # self.FBXActor.SyncToScene()

        # npz_timestamp = Time.TotalTime % self.NPZMotion.TotalTime
        # self.NPZActor.Skeleton.SetTransforms(
        #     Transform.TransformationFrom(
        #         self.NPZMotion.GetBoneTransformations(
        #             timestamps=npz_timestamp,
        #             bone_names_or_indices=list(self.NPZActor.Skeleton.Hierarchy.BoneNames),
        #             mirrored=self.Mirror,
        #         ),
        #         Transform.T(Vector3.Create(2, 0, 0))
        #     )
        # )
        # self.NPZActor.SyncToScene()

    def GUI(self):
        label_offset = Vector3.Create(0, 1, 0)
        AI4Animation.Draw.Text3D(
            ["BVH"],
            self.BVHActor.Skeleton.GetPositions(
                self.BVHActor.Skeleton.Hierarchy.GetBoneIndices(Definitions.HipName)
            )
            + label_offset,
            size=0.025,
            color=AI4Animation.Color.BLACK,
        )
        # AI4Animation.Draw.Text3D(
        #     ["FBX"],
        #     self.FBXActor.Skeleton.GetPositions(
        #         self.FBXActor.Skeleton.Hierarchy.GetBoneIndices(Definitions.HipName)
        #     ) + label_offset,
        #     size=0.025,
        #     color=AI4Animation.Color.BLACK,
        # )
        # AI4Animation.Draw.Text3D(
        #     ["NPZ"],
        #     self.NPZActor.Skeleton.GetPositions(
        #         self.NPZActor.Skeleton.Hierarchy.GetBoneIndices(Definitions.HipName)
        #     ) + label_offset,
        #     size=0.025,
        #     color=AI4Animation.Color.BLACK,
        # )


AI4Animation(Program(MOTION_FILE))
