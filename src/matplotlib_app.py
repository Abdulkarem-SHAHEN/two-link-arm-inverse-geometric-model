# matplotlib_app.py
# The same robot arm in a Matplotlib window, solved with a damped Newton's method that
# recovers from singular configurations. Click inside the outer circle to set the target.
#
# Run from the repository root:
#     python src/matplotlib_app.py

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use the Tkinter backend to avoid Qt warnings
import matplotlib.pyplot as plt

from kinematics import joint_positions, newton_ik


class App:
    """Draws the arm and handles mouse clicks."""

    def __init__(self):
        self.l1 = 1.0     # Length of first arm
        self.l2 = 1.0     # Length of second arm
        self.phi1 = 0.0   # Initial angle of first arm
        self.phi2 = 0.0   # Initial angle of second arm
        self.target = np.array([1.5, 0.5])  # Initial target position
        plt.ion()  # Interactive mode for animation
        self.fig, self.ax = plt.subplots()
        self.v0 = np.array([0.1, 0.1])  # Initial guess for joint angles
        self.draw()  # Draw initial state

        # Connect the mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def draw(self):
        # Clear the plot
        self.ax.clear()

        # Compute positions (base at the origin)
        x0, y0 = 0, 0
        (x1, y1), (x2, y2) = joint_positions(np.array([self.phi1, self.phi2]), self.l1, self.l2)

        # Draw the robot arms
        self.ax.plot([x0, x1, x2], [y0, y1, y2], 'b-', linewidth=2, label='Robot Arm')
        self.ax.plot(x0, y0, 'ko')  # Base
        self.ax.plot(x1, y1, 'ko')  # Elbow
        self.ax.plot(x2, y2, 'ko')  # End effector
        self.ax.plot(self.target[0], self.target[1], 'r*', label='Target')  # Target

        # Circles: path of the elbow (radius l1) and maximum reach (radius l1 + l2)
        circle1 = plt.Circle((0, 0), self.l1, fill=False, linestyle='--', color='gray')
        circle2 = plt.Circle((0, 0), self.l1 + self.l2, fill=False, linestyle='--', color='gray')
        self.ax.add_patch(circle1)
        self.ax.add_patch(circle2)

        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_aspect('equal')
        self.ax.legend()
        self.ax.grid(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def on_click(self, event):
        # Handle mouse click events to update the target position
        if event.inaxes != self.ax:
            return

        target_x, target_y = event.xdata, event.ydata

        # Check if target is within workspace
        target_dist = np.sqrt(target_x**2 + target_y**2)
        if target_dist > self.l1 + self.l2:
            print(f"Warning: Target ({target_x:.2f}, {target_y:.2f}) is outside the workspace!")
            return

        self.target = np.array([target_x, target_y])

        # Damped Newton's method, starting from the previous solution
        result = newton_ik(self.target, self.v0, self.l1, self.l2,
                           tol=1e-6, max_iter=100, damping=0.5, singular_eps=1e-10)
        self.phi1, self.phi2 = result.angles
        self.v0 = result.angles  # Initial guess for the next click

        # Redraw the robot with the new target
        self.draw()


if __name__ == "__main__":
    app = App()
    plt.ioff()
    plt.show()
