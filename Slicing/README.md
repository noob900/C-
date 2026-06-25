# Mesh Visualizer Guide

This README explains `meshvisual.py`: what the script is for, how the data moves through it, and what each function does.

The short version: the script loads a 3D mesh, slices it into horizontal layers, finds the outer contour of each layer, estimates surface normals, smooths those contours into toolpaths, converts points and normals into 6-DOF robot-style data, and displays everything in PyVista.

## How To Run

From the repository root:

```bash
.\.venv\Scripts\python.exe Slicing\meshvisual.py
```

The file currently loads:

```python
Slicing\15778_NoveltyBust_EgyptianPharaoh_V1_NEW.obj
```

Change the `mesh_path` value in `main()` if you want to visualize another mesh.

## Main Data Flow

1. `main()` creates a `MeshVisualizer`.
2. `MeshVisualizer.__init__()` stores settings and loads the mesh.
3. `generate_path_data()` calls `generate_layered_path()`.
4. `generate_layered_path()` slices the mesh layer by layer.
5. Each slice is turned into ordered contour points and normals.
6. `_smooth_layer()` tries to spline-smooth each contour.
7. `generate_frames()` creates visual coordinate frames.
8. `generate_6dof_data()` converts points and normals into XYZABC entries.
9. `get_scene()` builds the PyVista scene.
10. `.show()` opens the interactive viewer.

## Imports

```python
from dataclasses import dataclass
```

Gives the script `@dataclass`, which is used to make `LayeredPathData` a simple container class.

```python
import math
```

Used for trigonometry when converting normal vectors into roll, pitch, and yaw angles.

```python
from typing import Dict, List, Tuple
```

Used for type hints. These do not change runtime behavior, but they make the code easier to understand.

```python
import numpy as np
```

Used for vectors, matrices, point clouds, normals, and numerical operations.

```python
import pyvista as pv
```

Used to create the 3D interactive visualization window.

```python
from scipy.interpolate import splev, splprep
```

Used to make smooth spline curves from raw contour points.

```python
import trimesh
```

Used to load meshes, slice meshes, access face normals, and raycast against mesh surfaces.

## Constants

```python
EPSILON = 1e-9
```

A tiny number used to avoid division by zero when normalizing vectors.

```python
POINTS_PER_LAYER = 400
```

The number of sampled points generated for each smoothed layer path.

```python
MAX_FRAME_PREVIEW_COUNT = 100
```

Limits how many small coordinate frames are shown along the toolpath. This keeps the viewer from getting too crowded or slow.

## `LayeredPathData`

```python
@dataclass
class LayeredPathData:
```

A simple container for all generated path data.

It stores four lists:

- `smooth_paths`: green paths that were successfully spline-smoothed.
- `fallback_paths`: red paths that use raw points because smoothing failed.
- `smooth_normals`: normals matching the smooth paths.
- `fallback_normals`: normals matching the fallback paths.

### `points`

Combines `smooth_paths` and `fallback_paths` into one `N x 3` NumPy array.

### `normals`

Combines `smooth_normals` and `fallback_normals` into one `N x 3` NumPy array.

## Helper Functions

### `_stack_or_empty(parts)`

Takes a list of NumPy arrays and stacks the non-empty arrays into one large array.

If there is no data, it returns an empty array shaped like `0 x 3`.

### `_normalize(vector)`

Returns a unit-length version of a single vector.

Example:

```python
[10, 0, 0] -> [1, 0, 0]
```

### `_normalize_rows(vectors)`

Normalizes every row in a 2D array.

This is used when the script has many normals at once.

### `_tool_frame_axes(normal, ref_axis="world_x")`

Creates a full coordinate frame from one surface normal.

The normal becomes the local Z axis. Then the function chooses a stable reference direction and calculates matching X and Y axes.

Returns:

```python
x_axis, y_axis, z_axis
```

This function exists because the same frame-building logic is needed for both visualization frames and 6-DOF angle generation.

### `_normal_to_rpy(normal)`

Converts one surface normal into roll, pitch, and yaw angles in degrees.

It does this by:

1. Calling `_tool_frame_axes()` to build X/Y/Z axes.
2. Building a rotation matrix from those axes.
3. Converting the rotation matrix into Euler angles.

Returns:

```python
roll_degrees, pitch_degrees, yaw_degrees
```

### `_trimesh_to_pyvista(mesh)`

Converts a `trimesh.Trimesh` object into a `pyvista.PolyData` mesh.

PyVista wants faces in a slightly different format, so this function reshapes the face data before creating the PyVista object.

### `_line_polydata(segments)`

Converts a list of 3D line segments into one PyVista line object.

Each segment is expected to contain two points:

```python
[[x1, y1, z1], [x2, y2, z2]]
```

### `_axis_segments(origin, axes, length)`

Creates three line segments from one origin point.

Each line segment points along one axis and has the requested length.

### `_axis_meshes(origin, length, axes=None)`

Creates PyVista line meshes for red X, green Y, and blue Z axes.

Used for:

- The world coordinate frame.
- The mesh target-position frame.

### `_sample_frame_meshes(points, normals, length)`

Creates small RGB coordinate frames along sampled toolpath points.

For each point:

1. The matching normal is converted into X/Y/Z axes.
2. Three short axis lines are created.
3. The lines are grouped by color.

### `_add_line_meshes(plotter, meshes)`

Adds the prepared line meshes to a PyVista plotter.

This keeps `get_scene()` shorter and avoids repeating the same loop for every group of lines.

## Slicing And Path Generation Helpers

### `_outer_loop_points(section)`

Finds the outer contour of one sliced mesh section.

What it does:

1. Converts the 3D slice to a temporary 2D planar path.
2. Reads all closed loops from the slice.
3. Estimates the slice center.
4. Chooses the loop with the largest average distance from the center.
5. Converts that outer loop back to 3D points.

Returns either:

- an `N x 3` array of 3D contour points, or
- `None` if the section has no usable loop.

### `_fallback_normals(mesh, points)`

Creates simple radial normals when raycasting fails.

The fallback assumes each normal points outward from the mesh centroid in the XY plane.

### `_fill_missing_normals(normals)`

Some raycasts may fail for individual points. This function fills zero normals by copying the nearest non-zero neighbor normal.

After filling, it normalizes all normals again.

### `_raycast_normals(mesh, points, model_scale, z)`

Estimates surface normals for contour points.

What it does:

1. Finds the center of the current layer.
2. Builds horizontal rays starting slightly outside the contour.
3. Shoots rays inward toward the mesh.
4. Uses the hit triangle normals as point normals.
5. Fills missing normals if only some rays fail.
6. Uses `_fallback_normals()` if all rays fail.

The `z` value is only used for helpful warning messages.

### `_ordered_layer(points, normals, start_reference, forward)`

Reorders a layer so the toolpath starts near the previous layer's ending point.

It also alternates direction between layers. This makes the toolpath more continuous instead of jumping back to the same side every layer.

Returns:

```python
ordered_points, ordered_normals, layer_end_point
```

### `_smooth_layer(points, normals, smoothing)`

Attempts to create a smooth spline path through a layer.

It smooths both:

- XYZ positions
- IJK-style normal vectors

If the contour is closed and has enough points, it uses a periodic spline so the start and end connect cleanly.

Returns:

```python
smooth_points, smooth_normals
```

If SciPy cannot make the spline, the caller catches the exception and uses the raw points as a fallback.

### `generate_layered_path(mesh, num_layers=30, smoothing=0.1)`

This is the main mesh-processing function.

For each Z height:

1. Slice the mesh with `mesh.section()`.
2. Skip empty slices.
3. Extract the outer loop with `_outer_loop_points()`.
4. Estimate normals with `_raycast_normals()`.
5. Order the layer with `_ordered_layer()`.
6. Try to smooth the layer with `_smooth_layer()`.
7. Store smooth paths as green paths.
8. Store failed smooth paths as red fallback paths.

Returns a `LayeredPathData` object.

## `MeshVisualizer`

`MeshVisualizer` is the main class used by the script.

It owns:

- the loaded mesh
- generated points
- generated normals
- smoothed paths
- fallback paths
- 6-DOF toolpath data
- visualization frame meshes

### `__init__(...)`

Stores the configuration and immediately loads the mesh.

Important arguments:

- `mesh_path`: path to the `.obj`, `.stl`, or other mesh file.
- `wcs_origin`: world coordinate system origin.
- `stl_target_position`: where the mesh centroid should be moved.
- `num_layers`: how many horizontal slices to generate.
- `axis_length`: visual length of coordinate axes.

### `_load_and_position_mesh()`

Loads the mesh with Trimesh and moves its centroid to `stl_target_position`.

This makes the mesh position predictable in the viewer.

### `generate_path_data(smoothing=0.1)`

Calls `generate_layered_path()`, then stores the resulting paths, points, and normals on the class instance.

Run this before generating frames, 6-DOF data, or the final scene.

### `generate_frames()`

Creates visual coordinate frames.

It creates:

- a world frame at `wcs_origin`
- a mesh-position frame at `stl_target_position`
- sampled mini frames along the generated toolpath

### `generate_6dof_data()`

Converts every point and normal into an XYZABC dictionary.

Each output row looks like:

```python
{
    "X": x_position,
    "Y": y_position,
    "Z": z_position,
    "A": roll_degrees,
    "B": pitch_degrees,
    "C": yaw_degrees,
}
```

XYZ is relative to `wcs_origin`.

ABC is generated from the surface normal.

### `print_6dof_data_for_index(index)`

Prints one toolpath entry for quick inspection.

This is useful when you want to check whether the generated position and orientation values look reasonable.

### `get_scene()`

Creates and returns the PyVista plotter.

It adds:

1. The mesh.
2. Coordinate frames.
3. Green smoothed paths.
4. Red fallback paths.
5. Pickable black toolpath points.

The returned object is shown with:

```python
visualizer.get_scene().show()
```

### `_add_toolpaths(plotter)`

Adds green and red path lines to the PyVista scene.

Green means smoothing succeeded.

Red means the raw fallback path is being shown.

### `_add_pickable_points(plotter)`

Adds clickable toolpath points to the scene.

When you click a point, PyVista finds the closest toolpath point and displays its XYZABC data in the lower-right corner.

## `main()`

This function is the script entry point.

It:

1. Creates `MeshVisualizer`.
2. Generates path data.
3. Generates coordinate frames.
4. Generates 6-DOF data.
5. Prints one sample 6-DOF point.
6. Opens the PyVista viewer.

The last lines:

```python
if __name__ == "__main__":
    main()
```

mean: only run `main()` when this file is executed directly. If another file imports `meshvisual.py`, `main()` will not run automatically.

## Visual Meaning

- Light gray mesh: the loaded 3D model.
- Green paths: smoothed toolpaths.
- Red paths: fallback raw paths.
- Black dots: generated toolpath points.
- Yellow dot: currently picked point.
- Red axis: local X direction.
- Green axis: local Y direction.
- Blue axis: local Z direction, aligned with the surface normal for sampled frames.

## Common Changes

### Use a different mesh

Edit this line in `main()`:

```python
mesh_path=r"C:\Users\shish\C-\Slicing\15778_NoveltyBust_EgyptianPharaoh_V1_NEW.obj"
```

### Change slice density

Edit:

```python
num_layers=150
```

Higher values create more layers but take longer.

### Change path smoothness

Call:

```python
visualizer.generate_path_data(smoothing=0.05)
```

Lower values usually follow the raw contour more closely. Higher values smooth more aggressively.

### Show fewer preview frames

Edit:

```python
MAX_FRAME_PREVIEW_COUNT = 100
```

Lower values make the viewer less cluttered.

## Troubleshooting

### `ModuleNotFoundError`

Use the virtual environment:

```bash
.\.venv\Scripts\python.exe Slicing\meshvisual.py
```

### Red fallback paths appear

That means spline smoothing failed for that layer. The script still keeps the path by using raw contour points.

### Normal raycast warnings appear

That means the script could not raycast normals for a slice. It uses radial fallback normals instead.

### Viewer is slow

Try reducing:

- `num_layers`
- `POINTS_PER_LAYER`
- `MAX_FRAME_PREVIEW_COUNT`

