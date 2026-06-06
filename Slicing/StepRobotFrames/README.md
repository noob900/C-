# STEP Robot Frame Generator

This tool reads a STEP file, samples surface points across the model, builds a 6-axis tool frame at each point, and writes the frames to CSV.

The generated frame origin is the sampled surface point plus an optional stand-off distance along the surface normal. The frame Z axis points toward the surface by default, which is a common starting point for machining tool orientation.

## Install

This machine currently has the Windows Store `python.exe` shim, not a working Python install. Install Python 3.10+ first, then install the CAD dependency:

```powershell
cd C:\Users\shish\StepRobotFrames
python -m pip install -r requirements.txt
```

`cadquery` provides the OpenCascade `OCP` bindings used to read STEP geometry.

## Generate Frames

```powershell
cd C:\Users\shish\StepRobotFrames
python .\step_to_robot_frames.py C:\path\to\part.step --out frames.csv --samples-u 12 --samples-v 12
```

Useful options:

- `--samples-u` and `--samples-v`: surface grid density per face.
- `--max-points`: cap the total number of exported frames.
- `--stand-off`: offset frame origins away from the model along the surface normal.
- `--tool-axis toward_surface`: frame Z points into the model.
- `--tool-axis away_from_surface`: frame Z points out of the model.
- `--urscript robot_path.script`: optional Universal Robots style `movel(...)` export.
- `--unit-scale 0.001`: scale model units to robot units, for example millimeters to meters for URScript.

## CSV Output

Each row contains:

- sampled surface point in WCS
- surface normal in WCS
- frame X/Y/Z axes in WCS
- roll/pitch/yaw in degrees
- axis-angle rotation vector in radians

## Important

This generates geometric machining frames. It does not check robot reach, joint limits, collisions, singularities, fixture clearance, spindle/tool geometry, feeds, speeds, or controller-specific syntax. Run the output through your robot simulator/postprocessor before machining real material.
