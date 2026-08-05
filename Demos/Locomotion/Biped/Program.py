# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import raylib as rl

from ai4animation import (
    AI4Animation,
    Tensor,
    Time,
    Vector3,
)
from MotionController import MotionController, ControlParams

class Program:
    def __init__(
        self,
        prediction_fps=10
    ):
        self.PredictionFPS = prediction_fps

    def Start(self):
        self.MotionController = MotionController()
        self.ControlParams = None
        self.SetGuidance(0)

    def SetGuidance(self, index):
        self.GuidanceStyleIndex = index % len(self.MotionController.GuidanceNames)
        self.SelectedGuidanceName = self.MotionController.GuidanceNames[self.GuidanceStyleIndex]
        self.GuidancePose = self.MotionController.GuidanceTemplates[
            self.SelectedGuidanceName
        ].Positions.copy()
        if hasattr(self, "GuidanceDropdown"):
            self.GuidanceDropdown.Button.Label = f"Style: {self.SelectedGuidanceName}"

    def Update(self):
        if AI4Animation.IsStandalone():
            # Note: raylib is used for control inputs so headless/manual mode not yet supported
            speed_sprint = 2.0
            speed_normal = 1.0
            if AI4Animation.Standalone.IO.GamepadAvailable():
                left_stick = AI4Animation.Standalone.IO.GetLeftStick()
                right_stick = AI4Animation.Standalone.IO.GetRightStick()
                speed = (
                    speed_sprint
                    if AI4Animation.Standalone.IO.IsLeftStickPressed()
                    else speed_normal
                )

                # Handle guidance selection with L1 and R1 buttons
                if AI4Animation.Standalone.IO.IsL1Pressed():
                    self.SetGuidance(self.GuidanceStyleIndex - 1)
                if AI4Animation.Standalone.IO.IsR1Pressed():
                    self.SetGuidance(self.GuidanceStyleIndex + 1)

            # Keyboard control when no gamepad is available
            else:
                # WASD for left stick (velocity)
                # Left Shift for speed
                # Right Click and move mouse for right stick (direction)
                # Detail: We use momentum on the mouse direction start point to smooth out control over using GetMouseDeltaOnScreen() which is very noisy
                left_stick_vec3 = AI4Animation.Standalone.IO.GetWASDQE()
                left_stick = [left_stick_vec3[0], left_stick_vec3[2]]
                speed = speed_sprint if rl.IsKeyDown(rl.KEY_LEFT_SHIFT) else speed_normal
                if rl.IsMouseButtonDown(rl.MOUSE_BUTTON_RIGHT):
                    pos = np.array(AI4Animation.Standalone.IO.GetMousePositionOnScreen())
                    if self.DirectionMouseStart is None:
                        self.DirectionMouseStart = pos
                    else:
                        momentum = 0.01
                        self.DirectionMouseStart *= 1 - momentum
                        self.DirectionMouseStart += pos * momentum
                    right_stick = [
                        pos[0] - self.DirectionMouseStart[0],
                        self.DirectionMouseStart[1] - pos[1],
                    ]
                else:
                    self.DirectionMouseStart = None
                    right_stick = [0, 0]

                # Handle guidance selection with Q and E keys
                if rl.IsKeyPressed(rl.KEY_Q):
                    self.SetGuidance(self.GuidanceStyleIndex - 1)
                if rl.IsKeyPressed(rl.KEY_E):
                    self.SetGuidance(self.GuidanceStyleIndex + 1)

            velocity = speed * Vector3.ClampMagnitude(
                Vector3.Create(left_stick[0], 0, -left_stick[1]), 1.0
            )

            direction = Vector3.Create(right_stick[0], 0, -right_stick[1])

            position = Vector3.Lerp(
                self.MotionController.SimulationObject.GetPosition(0),
                self.MotionController.Actor.GetRootPosition(),
                self.MotionController.Synchronization,
            )
            self.ControlParams = ControlParams(
                position,
                direction,
                velocity,
                self.GuidancePose
            )

        if self.ControlParams is None:
            print("No control params have been set. Please assign the control params variable.")
            return

        self.MotionController.Update(
            self.ControlParams,
            Time.DeltaTime,
            self.PredictionFPS
        )

    def Standalone(self):
        AI4Animation.Standalone.IO.LogErrorIfGamepadNotAvailable()
        self.GuidanceStyleIndex = 0
        self.DirectionMouseStart = None
        AI4Animation.Standalone.Camera.SetTarget(self.MotionController.Actor.Entity)
        
        self.DrawRootControl = AI4Animation.GUI.Button(
            "Root Control", 0.8, 0.35, 0.15, 0.04, state=False
        )
        self.DrawGuidanceControl = AI4Animation.GUI.Button(
            "Guidance Control", 0.8, 0.40, 0.15, 0.04, state=False
        )
        self.DrawSequences = AI4Animation.GUI.Button(
            "Sequences", 0.8, 0.45, 0.15, 0.04, state=False
        )

        self.GuidanceDropdown = AI4Animation.GUI.Dropdown(
            f"Guidance: {self.SelectedGuidanceName}",
            0.375,
            0.1,
            0.25,
            0.04,
            options=[
                (name, (lambda _idx, i=i: self.SetGuidance(i)))
                for i, name in enumerate(self.MotionController.GuidanceNames)
            ],
        )

    def Draw(self):
        self.MotionController.SimulationObject.Draw()

        if self.DrawRootControl.Active:
            self.MotionController.RootControl.Draw()
        if self.DrawGuidanceControl.Active:
            self.MotionController.GuidanceControl.DrawLegacy(
                self.MotionController.Actor
            )
        if (
            self.DrawSequences.Active
            and self.MotionController.Previous is not None
            and self.MotionController.Sequence is not None
        ):
            self.MotionController.Previous.Draw(
                self.MotionController.Actor, AI4Animation.Color.RED
            )
            self.MotionController.Sequence.Draw(
                self.MotionController.Actor, AI4Animation.Color.GREEN
            )

    def GUI(self):
        if AI4Animation.Standalone.IO.GamepadAvailable():
            AI4Animation.Standalone.IO.DrawController(x=0.9, y=0.9, scale=0.5)
            AI4Animation.Draw.Text(
                "Left Stick: Move\nRight Stick: Facing Direction\nL1/R1: Change Style\nLeft Shift: Sprint",
                0.65,
                0.85,
                0.025,
                AI4Animation.Color.BLACK,
            )
        else:
            AI4Animation.Standalone.IO.DrawWASDQE(x=0.75, y=0.85, scale=0.5)
            AI4Animation.Draw.Text(
                "Gamepad recommended.",
                0.8,
                0.8,
                0.025,
                AI4Animation.Color.RED,
            )
            AI4Animation.Draw.Text(
                "WASD: Move\nShift: Sprint\nQ/E: Change Style\nRight Mouse Button: Facing Direction",
                0.865,
                0.85,
                0.025,
                AI4Animation.Color.BLACK,
            )

        AI4Animation.GUI.HorizontalBar(
            0.8,
            0.05,
            0.15,
            0.04,
            self.MotionController.Timescale,
            label="Timescale",
            limits=[1.0, self.MotionController.MaxTimescale],
        )
        AI4Animation.GUI.HorizontalBar(
            0.8,
            0.10,
            0.15,
            0.04,
            self.MotionController.Synchronization,
            label="Synchronization",
            limits=[0.0, 1.0],
        )
        if self.MotionController.Previous is not None:
            AI4Animation.GUI.HorizontalPivot(
                0.8,
                0.15,
                0.15,
                0.04,
                0.0,
                label="Previous Sequence",
                limits=[
                    self.MotionController.Previous.Timestamps[0],
                    self.MotionController.Previous.Timestamps[-1],
                ],
                pivotColor=AI4Animation.Color.RED,
            )
        if self.MotionController.Sequence is not None:
            AI4Animation.GUI.HorizontalPivot(
                0.8,
                0.20,
                0.15,
                0.04,
                0.0,
                label="Current Sequence",
                limits=[
                    self.MotionController.Sequence.Timestamps[0],
                    self.MotionController.Sequence.Timestamps[-1],
                ],
                pivotColor=AI4Animation.Color.GREEN,
            )
            AI4Animation.GUI.BarPlot(
                0.8,
                0.25,
                0.15,
                0.04,
                Tensor.SwapAxes(self.MotionController.Sequence.Contacts, 0, 1),
                label="Contacts",
            )

        self.DrawRootControl.GUI()
        self.DrawGuidanceControl.GUI()
        self.DrawSequences.GUI()

        self.GuidanceDropdown.Button.Label = f"Style: {self.SelectedGuidanceName}"
        self.GuidanceDropdown.GUI()


if __name__ == "__main__":
    AI4Animation(
        Program(
            prediction_fps=10
        ),
        mode=AI4Animation.Mode.STANDALONE
    )
