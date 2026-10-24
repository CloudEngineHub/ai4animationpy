# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Scene-independent skeleton with bone hierarchy, transforms, and velocities."""

import numpy as np
from ai4animation.Math import Quaternion, Rotation, Tensor, Transform, Vector3


class Skeleton:
    def __init__(self, hierarchy, root=None, transforms=None, velocities=None):
        self.Hierarchy = hierarchy
        bone_count = hierarchy.BoneCount

        self.ChildrenIndices = [[] for _ in range(bone_count)]
        self.SuccessorIndices = [[] for _ in range(bone_count)]
        for index, parent_index in enumerate(hierarchy.ParentIndices):
            if parent_index is None:
                continue
            if not hierarchy.IsValidBoneIndex(parent_index) or parent_index == index:
                continue
            self.ChildrenIndices[parent_index].append(index)

            ancestor = parent_index
            visited = set()
            while ancestor is not None:
                if ancestor in visited:
                    raise ValueError(
                        f"Hierarchy contains a cycle at bone index {ancestor}"
                    )
                visited.add(ancestor)
                self.SuccessorIndices[ancestor].append(index)
                ancestor = hierarchy.ParentIndices[ancestor]

        self.Root = root.copy() if root is not None else Transform.Identity()
        self.Transforms = (
            transforms.copy()
            if transforms is not None
            else Transform.Identity(bone_count)
        )
        self.Velocities = (
            velocities.copy() if velocities is not None else Vector3.Zero(bone_count)
        )
        parent_indices = np.array(hierarchy.GetParentIndices(), dtype=np.int64)
        sources = self.Transforms.copy()
        parents = self.Transforms[parent_indices].copy()
        root_mask = np.array(
            [parent_index is None for parent_index in hierarchy.ParentIndices]
        )
        singular_mask = np.linalg.matrix_rank(parents) < 4
        fallback_mask = root_mask | singular_mask
        sources[fallback_mask] = Transform.Identity()
        parents[fallback_mask] = Transform.Identity()
        self.ZeroTransforms = Transform.TransformationTo(sources, parents)

        singular_indices = np.flatnonzero(singular_mask & ~root_mask)
        if singular_indices.size:
            names = ", ".join(hierarchy.BoneNames[index] for index in singular_indices)
            print(
                "Warning: singular parent transforms for bones "
                f"[{names}]; using identity zero transforms."
            )

    def ComputeFK(self, local_transforms, root=None):
        """Compose local bone transforms into global transforms."""
        backend = Tensor.GetBackend(local_transforms)
        root = Tensor.ToBackend(
            self.Root if root is None else root,
            backend,
            reference=local_transforms,
        )
        local_transforms = Tensor.ToBackend(
            local_transforms, backend, reference=local_transforms
        )

        bone_count = self.Hierarchy.BoneCount
        if root.shape[-2:] != (4, 4):
            raise ValueError(f"Expected root shape (..., 4, 4), got {root.shape}")
        if local_transforms.shape[-3:] != (bone_count, 4, 4):
            raise ValueError(
                "Expected local_transforms shape "
                f"(..., {bone_count}, 4, 4), got {local_transforms.shape}"
            )

        try:
            batch_shape = Tensor.BroadcastShapes(
                root.shape[:-2], local_transforms.shape[:-3]
            )
        except ValueError as error:
            raise ValueError(
                "Root and local transform batch dimensions are not broadcastable: "
                f"{root.shape[:-2]} and {local_transforms.shape[:-3]}"
            ) from error

        root = Tensor.BroadcastTo(root, batch_shape + (4, 4))
        local_transforms = Tensor.BroadcastTo(
            local_transforms, batch_shape + (bone_count, 4, 4)
        )

        global_transforms = [None] * bone_count
        visiting = set()

        def ComputeTransform(index):
            transform = global_transforms[index]
            if transform is not None:
                return transform
            if index in visiting:
                raise ValueError(f"Hierarchy contains a cycle at bone index {index}")

            visiting.add(index)
            parent_index = self.Hierarchy.ParentIndices[index]
            parent = root if parent_index is None else ComputeTransform(parent_index)
            transform = Transform.TransformationFrom(
                local_transforms[..., index, :, :], parent
            )
            global_transforms[index] = transform
            visiting.remove(index)
            return transform

        for index in range(bone_count):
            ComputeTransform(index)

        return Tensor.Stack(global_transforms, axis=-3, backend=backend)

    def PrintSuccessors(self, index=0, indent=""):
        print(
            indent,
            self.Hierarchy.BoneNames[index],
            "->",
            [self.Hierarchy.BoneNames[item] for item in self.SuccessorIndices[index]],
        )
        for child_index in self.ChildrenIndices[index]:
            self.PrintSuccessors(child_index, indent + "  ")

    @staticmethod
    def _RequireSingleBoneSelection(names_or_indices, FK):
        if FK and not isinstance(names_or_indices, (int, str)):
            raise ValueError("FK assignments require a single bone")

    def SetTransforms(self, values, names_or_indices=None, FK=False):
        self._RequireSingleBoneSelection(names_or_indices, FK)
        index = self.Hierarchy.GetBoneIndices(names_or_indices)

        successors = self.SuccessorIndices[index] if FK else []
        if successors:
            local_transforms = Transform.TransformationTo(
                self.Transforms[successors], self.GetTransforms(index)
            )
        Transform.SetTransform(self.Transforms, values, index)
        if successors:
            self.Transforms[successors] = Transform.TransformationFrom(
                local_transforms, self.GetTransforms(index)
            )

    def GetTransforms(self, names_or_indices=None):
        index = self.Hierarchy.GetBoneIndices(names_or_indices)
        return Transform.GetTransform(self.Transforms, index)

    def SetPositionsAndRotations(
        self, positions, rotations, names_or_indices=None, FK=False
    ):
        self.SetTransforms(Transform.TR(positions, rotations), names_or_indices, FK=FK)

    def SetPositions(self, values, names_or_indices=None, FK=False):
        self._RequireSingleBoneSelection(names_or_indices, FK)
        if FK:
            transform = Tensor.Copy(self.GetTransforms(names_or_indices))
            Transform.SetPosition(transform, values)
            self.SetTransforms(transform, names_or_indices, FK=True)
            return

        index = self.Hierarchy.GetBoneIndices(names_or_indices)
        Transform.SetPosition(self.Transforms, values, index)

    def GetPositions(self, names_or_indices=None):
        index = self.Hierarchy.GetBoneIndices(names_or_indices)
        return Transform.GetPosition(self.Transforms, index)

    def SetRotations(self, values, names_or_indices=None, FK=False):
        self._RequireSingleBoneSelection(names_or_indices, FK)
        if FK:
            transform = Tensor.Copy(self.GetTransforms(names_or_indices))
            Transform.SetRotation(transform, values)
            self.SetTransforms(transform, names_or_indices, FK=True)
            return

        index = self.Hierarchy.GetBoneIndices(names_or_indices)
        Transform.SetRotation(self.Transforms, values, index)

    def GetRotations(self, names_or_indices=None):
        index = self.Hierarchy.GetBoneIndices(names_or_indices)
        return Transform.GetRotation(self.Transforms, index)

    def SetVelocities(self, values, names_or_indices=None):
        index = self.Hierarchy.GetBoneIndices(names_or_indices)
        Vector3.SetVector(self.Velocities, values, index)

    def GetVelocities(self, names_or_indices=None):
        index = self.Hierarchy.GetBoneIndices(names_or_indices)
        return Vector3.GetVector(self.Velocities, index)

    def GetRoot(self):
        return self.Root

    def SetRoot(self, root):
        self.Root = root

    def GetRootPosition(self):
        return Transform.GetPosition(self.Root)

    def GetRootRotation(self):
        return Transform.GetRotation(self.Root)

    def GetRootDirection(self):
        return Transform.GetAxisZ(self.Root)

    def RestoreZeroTransforms(
        self,
        names_or_indices=None,
        position=True,
        rotation=True,
        scale=1.0,
        FK=False,
    ):
        self._RequireSingleBoneSelection(names_or_indices, FK)
        scalar = isinstance(names_or_indices, (int, str))
        selection = self.Hierarchy.GetBoneIndices(names_or_indices)
        indices = selection if isinstance(selection, list) else [selection]
        indices = [
            index
            for index in indices
            if self.Hierarchy.ParentIndices[index] is not None
        ]
        if not indices:
            return

        selection = indices[0] if scalar else indices
        parent_indices = self.Hierarchy.GetParentIndices(indices)
        parent_selection = parent_indices[0] if scalar else parent_indices
        parent = self.GetTransforms(parent_selection)
        delta = Transform.TransformationTo(self.GetTransforms(selection), parent)
        if position:
            Transform.SetPosition(
                delta, scale * Transform.GetPosition(self.ZeroTransforms[selection])
            )
        if rotation:
            Transform.SetRotation(
                delta, Transform.GetRotation(self.ZeroTransforms[selection])
            )
        self.SetTransforms(
            Transform.TransformationFrom(delta, parent), selection, FK=FK
        )

    def GetDefaultBoneLengths(self, names_or_indices=None):
        selection = self.Hierarchy.GetBoneIndices(names_or_indices)
        transforms = Transform.GetTransform(self.ZeroTransforms, selection)
        return Vector3.Length(Transform.GetPosition(transforms))[..., 0]

    def GetCurrentBoneLengths(self, names_or_indices=None, parent_candidates=None):
        selection = self.Hierarchy.GetBoneIndices(names_or_indices)
        parents = self.Hierarchy.GetParentIndices(selection, parent_candidates)
        return Vector3.Distance(
            self.GetPositions(parents), self.GetPositions(selection)
        )[..., 0]

    # There are issues here if the learned bones and actual skeleton bones differ
    def RestoreBoneLengths(self, names_or_indices=None):
        selection = self.Hierarchy.GetBoneIndices(names_or_indices)
        self.SetBoneLengths(self.GetDefaultBoneLengths(selection), selection)

    # There are issues here if the learned bones and actual skeleton bones differ
    def SetBoneLengths(self, values, names_or_indices=None):
        selection = self.Hierarchy.GetBoneIndices(names_or_indices)
        parents = self.Hierarchy.GetParentIndices(selection)
        parent_positions = self.GetPositions(parents)
        positions = self.GetPositions(selection)
        directions = positions - parent_positions
        valid = Vector3.Length(directions) >= 1e-8
        fallback = Tensor.ZerosLike(directions)
        fallback[..., 0] = 1.0
        directions = Vector3.Normalize(Tensor.Where(valid, directions, fallback))
        lengths = Tensor.ToBackend(
            values, Tensor.GetBackend(parent_positions), reference=parent_positions
        ).reshape(-1, 1)
        positions = parent_positions + lengths * directions
        self.SetPositions(Tensor.Where(valid, positions, parent_positions), selection)

    @staticmethod
    def ComputeAlignment(source, from_pos, to_pos):
        source_position = Transform.GetPosition(source)
        from_direction = from_pos - source_position
        to_direction = to_pos - source_position
        valid = (Vector3.Length(from_direction) >= 1e-8) & (
            Vector3.Length(to_direction) >= 1e-8
        )
        fallback = Tensor.ZerosLike(from_direction)
        fallback[..., 0] = 1.0
        from_direction = Tensor.Where(valid, from_direction, fallback)
        to_direction = Tensor.Where(valid, to_direction, fallback)
        aligned = Rotation.Multiply(
            Quaternion.ToMatrix(Quaternion.FromTo(from_direction, to_direction)),
            Transform.GetRotation(source),
        )
        return Tensor.Where(valid[..., None], aligned, Transform.GetRotation(source))

    def RestoreBoneAlignments(self, names_or_indices=None):
        selection = self.Hierarchy.GetBoneIndices(names_or_indices)
        indices = selection if isinstance(selection, list) else [selection]
        indices = [index for index in indices if len(self.ChildrenIndices[index]) == 1]
        if not indices:
            return

        child_indices = [self.ChildrenIndices[index][0] for index in indices]
        transforms = self.GetTransforms(indices)
        self.SetRotations(
            self.ComputeAlignment(
                transforms,
                Vector3.PositionFrom(
                    Transform.GetPosition(self.ZeroTransforms[child_indices]),
                    transforms,
                ),
                self.GetPositions(child_indices),
            ),
            indices,
        )

    def CreateIK(self, source_name, target_name):
        """Create a FABRIK solver between two named bones."""
        # Import lazily because FABRIK depends on Skeleton's index-based API.
        from ai4animation.IK.FABRIK import FABRIK

        return FABRIK(self, source_name, target_name)
