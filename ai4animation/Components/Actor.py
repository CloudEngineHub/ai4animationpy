# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Skeletal actor component with model, scene entities, and visualization."""

from ai4animation import Utility
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Animation.Hierarchy import Hierarchy
from ai4animation.Animation.Skeleton import Skeleton
from ai4animation.Components.Component import Component


class Actor(Component):
    def Start(self, params):
        source_hierarchy = next((p for p in params if isinstance(p, Hierarchy)), None)
        self.ModelPath = next((p for p in params if isinstance(p, str)), None)
        bone_names = next((p for p in params if isinstance(p, (list, tuple))), None)

        if source_hierarchy is None and self.ModelPath is None:
            print("Error: Actor requires at least a Hierarchy or a model path.")
            return

        self.Model = None
        if self.ModelPath is not None:
            if self.ModelPath.lower().endswith(".fbx"):
                from ai4animation.Import.FBXImporter import FBX

                self.Model = FBX.Create(self.ModelPath)
            else:
                from ai4animation.Import.GLBImporter import GLB

                self.Model = GLB.Create(self.ModelPath)

        if source_hierarchy is not None:
            names, parents, transforms = (
                source_hierarchy.BoneNames,
                source_hierarchy.ParentNames,
                None,
            )
        else:
            names, parents, transforms = (
                self.Model.JointNames,
                self.Model.JointParents,
                self.Model.JointMatrices,
            )

        self.Entities = self.CreateEntities(names, parents, transforms)

        if source_hierarchy is not None and self.Model is not None:
            self.AssignZeroPose(self.Model.JointNames, self.Model.JointMatrices)

        selected = list(bone_names) if bone_names is not None else list(names)

        missing = [n for n in selected if n not in self.NameToEntity]
        if missing:
            raise ValueError(
                f"Actor: selected bones {missing} absent from scene entities "
                f"(full namespace has {len(self.NameToEntity)} entities)."
            )

        hierarchy = (
            source_hierarchy
            if source_hierarchy is not None
            else Hierarchy(names, parents)
        )

        self.Skeleton = Skeleton(
            Hierarchy(selected, hierarchy.FindParentNames(selected, selected)),
            root=self.Entity.GetTransform(),
            transforms=self.GetSceneTransforms(selected),
        )

    def Update(self):
        pass

    def CreateCopy(self, name=None):
        entity = AI4Animation.Scene.AddEntity(
            self.Entity.Name if name is None else name
        )
        if self.ModelPath is not None:
            return entity.AddComponent(
                Actor, self.ModelPath, self.Skeleton.Hierarchy.BoneNames
            )
        return entity.AddComponent(Actor, self.Skeleton.Hierarchy)

    def GetBoneEntity(self, name):
        entity = self.NameToEntity.get(name)
        if entity is None:
            print(f"Entity with name '{name}' could not be found.")
        return entity

    def GetBoneEntities(self, names=None):
        if names is None:
            names = list(self.Skeleton.Hierarchy.BoneNames)
        return [self.NameToEntity[name] for name in names]

    def GetBoneEntityIndices(self, names=None):
        if names is None:
            names = list(self.Skeleton.Hierarchy.BoneNames)
        return [self.NameToEntity[name].Index for name in names]

    def GetSceneBoneNames(self):
        return [entity.Name for entity in self.Entities]

    def GetSceneParentNames(self):
        return [
            entity.Parent.Name if entity.Parent is not self.Entity else None
            for entity in self.Entities
        ]

    def SetSceneTransforms(self, values, names=None):
        indices = (
            self.GetBoneEntityIndices(names)
            if names is not None
            else [entity.Index for entity in self.Entities]
        )
        AI4Animation.Scene.Transforms[indices] = values

    def GetSceneTransforms(self, names=None):
        indices = (
            self.GetBoneEntityIndices(names)
            if names is not None
            else [entity.Index for entity in self.Entities]
        )
        return AI4Animation.Scene.Transforms[indices]

    def SetScale(self, value, fk=True):
        self.Entity.SetScale(value, fk=fk)

    def GetScale(self):
        return self.Entity.GetScale()

    def SyncToScene(self, bones=None, root=True):
        if root:
            self.Entity.SetTransform(self.Skeleton.Root)
        for index in self.Skeleton.Hierarchy.GetBoneIndices(bones):
            if not self.Skeleton.Hierarchy.IsValidBoneIndex(index):
                continue
            entity = self.NameToEntity.get(self.Skeleton.Hierarchy.BoneNames[index])
            if entity is not None:
                entity.SetTransform(self.Skeleton.GetTransforms(index))

    def SyncFromScene(self, bones=None, root=True):
        if root:
            self.Skeleton.SetRoot(self.Entity.GetTransform())
        for index in self.Skeleton.Hierarchy.GetBoneIndices(bones):
            if not self.Skeleton.Hierarchy.IsValidBoneIndex(index):
                continue
            entity = self.NameToEntity.get(self.Skeleton.Hierarchy.BoneNames[index])
            if entity is not None:
                self.Skeleton.SetTransforms(entity.GetTransform(), index)

    def AssignZeroPose(self, names, Transforms, entity=None):
        if entity is None:
            entity = self.Entity

        name_to_index = {n: i for i, n in enumerate(names)}
        stack = [entity]
        while stack:
            ent = stack.pop()
            idx = name_to_index.get(ent.Name)
            if idx is not None:
                AI4Animation.Scene.Transforms[ent.Index] = Transforms[idx]
            stack.extend(ent.Children)

    def CreateEntities(self, names, parents, transforms=None):
        self.NameToEntity = {
            name: AI4Animation.Scene.AddEntity(
                name, position=None, rotation=None, parent=self.Entity
            )
            for name in names
        }
        entities = list(self.NameToEntity.values())

        for name, parent_name in zip(names, parents):
            parent = self.NameToEntity.get(parent_name)
            if parent is not None:
                self.NameToEntity[name].SetParent(parent)

        if transforms is not None:
            self.AssignZeroPose(names, transforms, self.Entity)

        return entities

    def DrawHandle(self):
        self.Entity.DrawHandle()

    def DrawHandles(self):
        for index in range(self.Skeleton.Hierarchy.BoneCount):
            entity = self.NameToEntity.get(self.Skeleton.Hierarchy.BoneNames[index])
            if entity is not None:
                entity.DrawHandle()

    def ShowMesh(self, value):
        if self.SkinnedMesh is None:
            return
        if value:
            self.SkinnedMesh.Register()
        else:
            if AI4Animation.Standalone.RenderPipeline.HasModel(self.SkinnedMesh.Models):
                self.SkinnedMesh.Unregister()

    def ToggleMesh(self):
        if self.SkinnedMesh is None:
            return
        if AI4Animation.Standalone.RenderPipeline.HasModel(self.SkinnedMesh.Models):
            self.SkinnedMesh.Unregister()
        else:
            self.SkinnedMesh.Register()

    def Standalone(self):
        self.SkinnedMesh = None
        self.Button_Mesh = None

        if self.Model is not None:
            self.SkinnedMesh = AI4Animation.Standalone.CreateSkinnedMesh(
                self, self.Model
            )

        self.Canvas = AI4Animation.GUI.Canvas("Actor", 0.01, 0.3, 0.125, 0.25)
        tuples = [
            ("Button_Root", "Show Root"),
            ("Button_Skeleton", "Show Skeleton"),
            ("Button_Velocities", "Show Velocities"),
        ]
        if self.Model is not None:
            tuples.append(("Button_Mesh", "Show Mesh"))
        tuples.extend(
            [
                ("Button_Labels", "Show Labels"),
                ("Button_Hierarchy", "Show Hierarchy"),
            ]
        )

        for i, (button, text) in enumerate(tuples):
            setattr(
                self,
                button,
                AI4Animation.GUI.Button(
                    text,
                    0.05,
                    0.15 + i / (len(tuples) + 1),
                    0.9,
                    0.65 / len(tuples),
                    False,
                    True,
                    self.Canvas,
                ),
            )

        if self.Button_Mesh is not None:
            self.Button_Mesh.Active = True

    def Draw(self):
        boneSize = 0.0175
        if self.Button_Root.Active:
            AI4Animation.Draw.Transform(self.Skeleton.Root, 0.5)
        if self.Button_Skeleton.Active:
            AI4Animation.Draw.Transform(self.Skeleton.Transforms, 0.25)
            AI4Animation.Draw.Cylinder(
                self.Skeleton.GetPositions(self.Skeleton.Hierarchy.GetParentIndices()),
                self.Skeleton.GetPositions(self.Skeleton.Hierarchy.GetBoneIndices()),
                boneSize,
                0.0,
                10,
                Utility.Opacity(AI4Animation.Color.BLACK, 0.75),
            )
        if self.Button_Velocities.Active:
            AI4Animation.Draw.Vector(
                self.Skeleton.GetPositions(),
                self.Skeleton.Velocities,
                boneSize,
                Utility.Opacity(AI4Animation.Color.BLUE, 0.5),
            )
        if self.Button_Hierarchy.Active:
            self.Entity.DrawHierarchy()

    def GUI(self):
        self.Canvas.GUI()
        self.Button_Root.GUI()
        self.Button_Skeleton.GUI()
        self.Button_Velocities.GUI()
        if self.Button_Mesh is not None:
            self.Button_Mesh.GUI()
        self.Button_Labels.GUI()
        self.Button_Hierarchy.GUI()

        if self.Button_Labels.Active:
            AI4Animation.Draw.Text3D(
                list(self.Skeleton.Hierarchy.BoneNames),
                self.Skeleton.GetPositions(),
                0.01,
                AI4Animation.Color.BLACK,
            )

        if self.Button_Mesh is not None and self.Button_Mesh.IsPressed():
            self.ToggleMesh()
