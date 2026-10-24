# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import TYPE_CHECKING

from ai4animation import Rotation, Vector3

if TYPE_CHECKING:
    from ai4animation.IK.FABRIK import FABRIK


class LegIK:
    def __init__(self, ik: "FABRIK"):
        self.IK = ik
        ee_pos = ik.Skeleton.GetPositions(ik.LastBoneIndex()).copy()
        self.EEBaseline: float = ee_pos[..., 1]

        self.TargetPosition = Vector3.Create(0, 0, 0)
        self.TargetRotation = Rotation.Euler(0, 0, 0)

        self.TargetPosition = self.IK.Skeleton.GetPositions(self.IK.LastBoneIndex())
        self.TargetRotation = self.IK.Skeleton.GetRotations(self.IK.LastBoneIndex())

    def Solve(
        self,
        contact: float,
        maxIterations: int,
        maxAccuracy: float,
    ):
        self.TargetPosition = Vector3.Lerp(
            self.IK.Skeleton.GetPositions(self.IK.LastBoneIndex()),
            self.TargetPosition,
            contact,
        )
        self.TargetRotation = self.IK.Skeleton.GetRotations(self.IK.LastBoneIndex())
        self.IK.Solve(
            self.TargetPosition,
            self.TargetRotation,
            maxIterations,
            maxAccuracy,
        )
