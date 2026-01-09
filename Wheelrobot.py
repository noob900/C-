import numpy as np
import matplotlib.pyplot as plt

class FourWheeledRobot:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, width=0.4, length=0.6):
        self.x = x          # Center X position
        self.y = y          # Center Y position
        self.yaw = yaw      # Rotation in radians
        self.width = width  # Chassis width
        self.length = length # Chassis length
        
        # Wheel dimensions
        self.wheel_w = 0.1
        self.wheel_l = 0.2

    def get_rotation_matrix(self):
        return np.array([
            [np.cos(self.yaw), -np.sin(self.yaw)],
            [np.sin(self.yaw),  np.cos(self.yaw)]
        ])

    def draw(self, ax):
        rot = self.get_rotation_matrix()
        
        # 1. Define Chassis (4 corners relative to center)
        chassis_outline = np.array([
            [-self.length/2,  self.width/2],
            [ self.length/2,  self.width/2],
            [ self.length/2, -self.width/2],
            [-self.length/2, -self.width/2],
            [-self.length/2,  self.width/2]
        ]).T
        
        # 2. Define 4 Wheel Positions (Relative to center)
        # Front-Left, Front-Right, Rear-Left, Rear-Right
        wheel_offsets = [
            [ self.length/2.5,  self.width/1.8],
            [ self.length/2.5, -self.width/1.8],
            [-self.length/2.5,  self.width/1.8],
            [-self.length/2.5, -self.width/1.8]
        ]

        # Rotate and translate chassis
        chassis_plot = rot @ chassis_outline + np.array([[self.x], [self.y]])
        ax.plot(chassis_plot[0, :], chassis_plot[1, :], "k-", lw=2) # Draw body
        
        # Draw each wheel
        for offset in wheel_offsets:
            wheel_box = np.array([
                [-self.wheel_l/2,  self.wheel_w/2],
                [ self.wheel_l/2,  self.wheel_w/2],
                [ self.wheel_l/2, -self.wheel_w/2],
                [-self.wheel_l/2, -self.wheel_w/2],
                [-self.wheel_l/2,  self.wheel_w/2]
            ]).T
            # Rotate wheel box, add its offset, then rotate by robot's yaw
            wheel_plot = rot @ (wheel_box + np.array([[offset[0]], [offset[1]]])) + np.array([[self.x], [self.y]])
            ax.fill(wheel_plot[0, :], wheel_plot[1, :], "darkgrey", alpha=0.8)

        # Direction arrow (Front)
        arrow = rot @ np.array([[0, self.length/2], [0, 0]]) + np.array([[self.x], [self.y]])
        ax.arrow(self.x, self.y, (arrow[0,1]-self.x)*0.8, (arrow[1,1]-self.y)*0.8, 
                 head_width=0.05, head_length=0.1, fc='r', ec='r')

# --- Simulation Setup ---
plt.figure(figsize=(8,8))
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.grid(True)

# Create robot at (0,0) tilted 45 degrees
robot = FourWheeledRobot(x=0.5, y=0.5, yaw=np.radians(45))
robot.draw(ax)

plt.title("Top View: 4-Wheeled Autonomous Robot Simulation")
plt.show()