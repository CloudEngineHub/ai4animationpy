# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Skeleton hierarchy: bone names, parent names, and index mappings."""


class Hierarchy:
    def __init__(self, bone_names, parent_names):
        self.BoneNames = bone_names
        self.BoneCount = len(bone_names)
        self.ParentNames = parent_names
        self.NameToIndex = {name: i for i, name in enumerate(bone_names)}

        # Convert parent names to parent indices
        self.ParentIndices = []
        for parent_name in parent_names:
            if parent_name is None:
                self.ParentIndices.append(None)  # Root bone
            else:
                parent_idx = self.NameToIndex.get(parent_name)
                self.ParentIndices.append(parent_idx)

    def GetBoneIndices(self, names_or_indices=None):
        if names_or_indices is None:
            return list(range(self.BoneCount))

        def Resolve(value):
            if isinstance(value, int):
                if not self.IsValidBoneIndex(value):
                    raise IndexError(f"Bone index {value} is out of range")
                return value
            if not isinstance(value, str):
                raise TypeError("Bone selections must be names, indices, or lists")
            try:
                return self.NameToIndex[value]
            except KeyError as error:
                raise KeyError(f"Bone '{value}' not found") from error

        if isinstance(names_or_indices, list):
            return [Resolve(value) for value in names_or_indices]
        return Resolve(names_or_indices)

    def GetParentIndices(self, names_or_indices=None, candidates=None):
        indices = self.GetBoneIndices(names_or_indices)
        candidate_indices = (
            None if candidates is None else set(self.GetBoneIndices(candidates))
        )

        def GetParent(index):
            if index is None:
                return None

            parent_index = self.ParentIndices[index]
            if candidate_indices is None:
                return index if parent_index is None else parent_index

            while parent_index is not None:
                if parent_index in candidate_indices:
                    return parent_index
                parent_index = self.ParentIndices[parent_index]
            return index

        if isinstance(indices, list):
            return [GetParent(index) for index in indices]
        return GetParent(indices)

    def GetChainIndices(self, source, target):
        if not isinstance(source, (int, str)) or not isinstance(target, (int, str)):
            raise TypeError("IK chains must use scalar bone names or indices")

        source_index = self.GetBoneIndices(source)
        target_index = self.GetBoneIndices(target)

        chain = [target_index]
        pivot = target_index
        while pivot != source_index:
            pivot = self.ParentIndices[pivot]
            if pivot is None:
                raise ValueError(f"Bone '{target}' is not a descendant of '{source}'")
            chain.append(pivot)
        chain.reverse()
        return chain

    def FindParentNames(self, bones, candidates):
        candidate_set = set(candidates)
        parents = []
        for bone in bones:
            idx = self.NameToIndex.get(bone)
            parent = None
            if self.IsValidBoneIndex(idx):
                idx = self.ParentIndices[idx]
                while self.IsValidBoneIndex(idx):
                    if self.BoneNames[idx] in candidate_set:
                        parent = self.BoneNames[idx]
                        break
                    idx = self.ParentIndices[idx]
            parents.append(parent)
        return parents

    def IsValidBoneIndex(self, index):
        return index is not None and 0 <= index < len(self.BoneNames)

    def Debug(self):
        print("=== Hierarchy ===")
        print(f"Bones: {len(self.BoneNames)}")
        for i, name in enumerate(self.BoneNames):
            parent = self.ParentNames[i] if self.ParentNames[i] is not None else "None"
            print(
                f"  [{i}] {name} (parent: {parent}) (parent_idx: {self.ParentIndices[i]})"
            )
