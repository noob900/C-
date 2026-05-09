# Robotics Projects

This folder contains robotics simulations, control systems, and autonomous navigation implementations using PyBullet physics engine.

## 📁 Contents

### Core Projects
- **quadbot.py** - Main quadruped robot simulation with custom URDF control interface
- **quadbot_custom.urdf** - Custom URDF definition for the quadruped robot (3 joints per leg)
- **Wheelrobot.py** - Wheeled robot simulation and control
- **Autonomous.py** - Autonomous navigation and path planning implementation

### Control Systems
- **joystick.py** - Joystick/gamepad control interface for robots
- **RRTbase.py** - Rapidly-exploring Random Tree (RRT) path planning algorithm

### Hardware Projects
- **Sorting-Station-PLC/** - PLC control code for industrial sorting station

## 🚀 Quick Start

### 🛠 Visual Development Workflow
1. **Open the URDF Preview**: Open `quadbot_custom.urdf`, press `Ctrl+Shift+P`, and run `URDF: Preview`.
2. **Run the Simulation**:
```bash
python quadbot.py
```
This launches the PyBullet GUI where you can control the robot using sliders.

### Run Wheelrobot
```bash
python Wheelrobot.py
```

### Use Joystick Control
```bash
python joystick.py
```

## 📋 Dependencies
- pybullet
- numpy
- pygame (for joystick support)

## 🔧 How to Modify

### Edit Robot URDF
- Open `quadbot_custom.urdf` and modify link sizes, masses, or joint limits
- Reload the simulation to see changes

### Add New Joints
- Duplicate a leg block in the URDF file
- Update joint names and positions
- Add corresponding control logic in Python script

## 📝 Notes
- Joystick requires gamepad/controller connected to system
- RRT algorithm useful for obstacle avoidance paths
- All simulations use PyBullet's built-in physics engine
