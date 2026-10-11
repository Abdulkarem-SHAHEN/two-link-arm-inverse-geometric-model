# tkinter_app.py
# Interactive two-link robot arm on a Tkinter canvas. Click anywhere: the arm moves its
# end effector to the clicked point.
#
# Run from the repository root:
#     python src/tkinter_app.py                    # closed-form solver (default)
#     python src/tkinter_app.py --solver newton    # Newton's method

import argparse
import tkinter as tk

import numpy as np

from kinematics import analytic_ik, forward_kinematics, joint_positions, newton_ik


class TwoArmRobotApp:
    def __init__(self, master, solver="analytic", L1=150, L2=100, tol=1e-3, max_iter=20):
        self.master = master
        self.solver = solver
        self.L1 = L1
        self.L2 = L2
        self.tol = tol
        self.max_iter = max_iter

        self.width = 600
        self.height = 600
        self.center = np.array([self.width // 2, self.height // 2], dtype=float)

        # Initial joint angles; the first target is where the arm already is
        self.angles = np.array([0.5, 0.5])
        self.target = forward_kinematics(self.angles, self.L1, self.L2)

        # Canvas setup
        self.canvas = tk.Canvas(master, width=self.width, height=self.height, bg='white')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        # First draw
        self.draw()

    def on_click(self, event):
        # Canvas coordinates -> coordinates relative to the base of the arm.
        # The canvas y-axis points down, so positive angles turn clockwise on screen.
        click = np.array([event.x, event.y], dtype=float)
        self.target = click - self.center

        if self.solver == "newton":
            # Start from the current pose: the nearest solution is found in a few iterations
            result = newton_ik(self.target, self.angles, self.L1, self.L2,
                               tol=self.tol, max_iter=self.max_iter)
        else:
            result = analytic_ik(self.target, self.angles, self.L1, self.L2)

        self.angles = result.angles
        self.draw()

    def draw(self):
        self.canvas.delete("all")
        cx, cy = self.center

        # Circles: path of the elbow (radius L1) and maximum reach (radius L1 + L2)
        for r in (self.L1, self.L1 + self.L2):
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline='#ccc')

        # Base joint
        rj = 8
        self.canvas.create_oval(cx - rj, cy - rj, cx + rj, cy + rj, fill='black')

        # Compute joint positions
        p1, p2 = joint_positions(self.angles, self.L1, self.L2)

        # First arm
        self.canvas.create_line(cx, cy, cx + p1[0], cy + p1[1], width=6, fill='blue')
        self.canvas.create_oval(cx + p1[0] - rj, cy + p1[1] - rj,
                                cx + p1[0] + rj, cy + p1[1] + rj, fill='black')

        # Second arm
        self.canvas.create_line(cx + p1[0], cy + p1[1],
                                cx + p2[0], cy + p2[1], width=6, fill='blue')
        self.canvas.create_oval(cx + p2[0] - rj, cy + p2[1] - rj,
                                cx + p2[0] + rj, cy + p2[1] + rj, fill='black')

        # Target
        tx, ty = self.target + self.center
        rt = 5
        self.canvas.create_oval(tx - rt, ty - rt, tx + rt, ty + rt, fill='red')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-link robot arm: click to set the target.")
    parser.add_argument("--solver", choices=("analytic", "newton"), default="analytic",
                        help="inverse kinematics method (default: analytic)")
    args = parser.parse_args()

    root = tk.Tk()
    root.title(f"Two-Arm Robot Inverse Kinematics ({args.solver})")
    app = TwoArmRobotApp(root, solver=args.solver)
    root.mainloop()
