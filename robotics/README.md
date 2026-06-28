# Robotics Projects

This folder contains robotics simulations, control-system experiments, and autonomous navigation work, mainly using PyBullet.

## Contents

### Core Projects

- `quadbot.py` - Main quadruped robot simulation with custom URDF control.
- `quadbot_custom.urdf` - Custom quadruped robot definition.
- `Wheelrobot.py` - Wheeled robot simulation and control.
- `Autonomous.py` - Autonomous navigation and path-planning experiments.
- `Autonomous2.py` - Maze-based autonomous search using LIDAR-style sensing without path planning.

### Control and Planning

- `joystick.py` - Joystick/gamepad control interface.
- `RRTbase.py` - Rapidly-exploring Random Tree path-planning algorithm.
- `Robot_Stability.py` - Robot stability experiments.

### Supporting Files

- `Drone.png` - Drone-related image or diagram.
- `GalatAnswer.html` - HTML report or generated project document.

## Quick Start

Run the quadruped simulation:

```bash
python quadbot.py
```

Run the wheeled robot simulation:

```bash
python Wheelrobot.py
```

Run joystick control:

```bash
python joystick.py
```

Run the maze LIDAR search simulation:

```bash
python robotics/Autonomous2.py
```

## Dependencies

- `pybullet`
- `numpy`
- `pygame` for joystick support

## Notes

- Joystick scripts require a connected gamepad/controller.
- RRT is useful for obstacle-avoidance path experiments.
- PyBullet GUI scripts should be run from a local desktop session.
