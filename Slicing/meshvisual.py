from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv
from scipy.interpolate import splev, splprep
import trimesh


EPSILON = 1e-9
POINTS_PER_LAYER = 400
MAX_FRAME_PREVIEW_COUNT = 100


@dataclass
class LayeredPathData:
    smooth_paths: List[np.ndarray]
    fallback_paths: List[np.ndarray]
    smooth_normals: List[np.ndarray]
    fallback_normals: List[np.ndarray]

    @property
    def points(self) -> np.ndarray:
        return _stack_or_empty(self.smooth_paths + self.fallback_paths)

    @property
    def normals(self) -> np.ndarray:
        return _stack_or_empty(self.smooth_normals + self.fallback_normals)


def _stack_or_empty(parts: List[np.ndarray]) -> np.ndarray:
    valid_parts = [part for part in parts if len(part) > 0]
    return np.vstack(valid_parts) if valid_parts else np.empty((0, 3))


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + EPSILON)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPSILON)


def _tool_frame_axes(normal: np.ndarray, ref_axis: str = "world_x") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = _normalize(normal)
    references = {
        "world_x": np.array([1.0, 0.0, 0.0]),
        "world_y": np.array([0.0, 1.0, 0.0]),
        "world_z": np.array([0.0, 0.0, 1.0]),
    }
    ref = references.get(ref_axis, references["world_x"])

    if abs(np.dot(ref, z_axis)) > 0.95:
        ref = references["world_y"]
    if abs(np.dot(ref, z_axis)) > 0.95:
        ref = references["world_z"]

    x_axis = _normalize(ref - z_axis * np.dot(ref, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    return x_axis, y_axis, z_axis


def _normal_to_rpy(normal: np.ndarray) -> Tuple[float, float, float]:
    x_axis, y_axis, z_axis = _tool_frame_axes(normal)
    rotation = np.array([x_axis, y_axis, z_axis]).T

    r20 = rotation[2, 0]
    if abs(r20) < 1.0 - EPSILON:
        pitch = math.asin(-r20)
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        pitch = math.pi / 2.0 if r20 <= -1.0 else -math.pi / 2.0
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _trimesh_to_pyvista(mesh: trimesh.Trimesh) -> pv.PolyData:
    if mesh is None or mesh.is_empty:
        return pv.PolyData()

    faces = np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces))
    return pv.PolyData(mesh.vertices, faces)


def _line_polydata(segments: List[np.ndarray]) -> pv.PolyData:
    if not segments:
        return pv.PolyData()

    points = np.asarray(segments, dtype=float).reshape(-1, 3)
    lines = []
    for index in range(0, len(points), 2):
        lines.extend([2, index, index + 1])
    return pv.PolyData(points, lines=np.array(lines))


def _axis_segments(origin: np.ndarray, axes: Tuple[np.ndarray, np.ndarray, np.ndarray], length: float) -> List[np.ndarray]:
    origin = np.asarray(origin, dtype=float)
    return [np.array([origin, origin + axis * length]) for axis in axes]


def _axis_meshes(origin: np.ndarray, length: float, axes=None) -> List[Tuple[pv.PolyData, str, int]]:
    if axes is None:
        axes = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

    colors = ["red", "green", "blue"]
    return [
        (_line_polydata([segment]), color, 3)
        for segment, color in zip(_axis_segments(origin, axes, length), colors)
    ]


def _sample_frame_meshes(points: np.ndarray, normals: np.ndarray, length: float) -> List[Tuple[pv.PolyData, str, int]]:
    x_segments, y_segments, z_segments = [], [], []
    for point, normal in zip(points, normals):
        x_axis, y_axis, z_axis = _tool_frame_axes(normal)
        x_segments.append(np.array([point, point + x_axis * length]))
        y_segments.append(np.array([point, point + y_axis * length]))
        z_segments.append(np.array([point, point + z_axis * length]))

    return [
        (_line_polydata(x_segments), "red", 1),
        (_line_polydata(y_segments), "green", 1),
        (_line_polydata(z_segments), "blue", 1),
    ]


def _add_line_meshes(plotter: pv.Plotter, meshes: List[Tuple[pv.PolyData, str, int]]) -> None:
    for mesh, color, width in meshes:
        if mesh.n_points > 0:
            plotter.add_mesh(mesh, color=color, line_width=width)


def _outer_loop_points(section) -> np.ndarray | None:
    planar_path, to_3d = section.to_2D()
    loops = [np.asarray(loop) for loop in planar_path.discrete if len(loop) >= 3]
    if not loops:
        return None

    centroid = np.mean(np.vstack(loops), axis=0)
    outer_loop = max(loops, key=lambda loop: np.mean(np.linalg.norm(loop - centroid, axis=1)))
    z_values = np.zeros(len(outer_loop))
    return trimesh.transform_points(np.column_stack((outer_loop, z_values)), to_3d)


def _fallback_normals(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    radial = points - mesh.centroid
    radial[:, 2] = 0.0
    normals = _normalize_rows(radial)
    zero_rows = np.linalg.norm(normals, axis=1) < EPSILON
    normals[zero_rows] = [0.0, 0.0, 1.0]
    return normals


def _fill_missing_normals(normals: np.ndarray) -> np.ndarray:
    filled = normals.copy()
    for index, normal in enumerate(filled):
        if not np.allclose(normal, 0.0):
            continue

        for offset in range(1, len(filled)):
            previous = (index - offset) % len(filled)
            next_index = (index + offset) % len(filled)
            if not np.allclose(filled[previous], 0.0):
                filled[index] = filled[previous]
                break
            if not np.allclose(filled[next_index], 0.0):
                filled[index] = filled[next_index]
                break

    return _normalize_rows(filled)


def _raycast_normals(mesh: trimesh.Trimesh, points: np.ndarray, model_scale: float, z: float) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    radial_xy = points[:, :2] - centroid[:2]
    radial_xy = radial_xy / (np.linalg.norm(radial_xy, axis=1, keepdims=True) + EPSILON)

    origins = points.copy()
    origins[:, :2] += radial_xy * model_scale * 0.01
    directions = np.column_stack((-radial_xy, np.zeros(len(points))))

    _, ray_indices, triangle_indices = mesh.ray.intersects_location(origins, directions)
    if len(ray_indices) == 0:
        print(f"Warning: normal raycast failed at Z={z:.2f}; using radial fallback.")
        return _fallback_normals(mesh, points)

    normals = np.zeros_like(points)
    hit_normals = mesh.face_normals[triangle_indices]
    unique_rays, first_hits = np.unique(ray_indices, return_index=True)
    normals[unique_rays] = hit_normals[first_hits]
    return _fill_missing_normals(normals)


def _ordered_layer(
    points: np.ndarray,
    normals: np.ndarray,
    start_reference: np.ndarray | None,
    forward: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = points[0] if start_reference is None else start_reference
    start_index = np.argmin(np.linalg.norm(points - reference, axis=1))

    ordered_points = np.roll(points, -start_index, axis=0)
    ordered_normals = np.roll(normals, -start_index, axis=0)
    if not forward:
        ordered_points = np.flip(ordered_points, axis=0)
        ordered_normals = np.flip(ordered_normals, axis=0)

    unique_mask = np.ones(len(ordered_points), dtype=bool)
    unique_mask[1:] = np.linalg.norm(np.diff(ordered_points, axis=0), axis=1) > 1e-4
    return ordered_points[unique_mask], ordered_normals[unique_mask], ordered_points[-1]


def _smooth_layer(points: np.ndarray, normals: np.ndarray, smoothing: float) -> Tuple[np.ndarray, np.ndarray]:
    is_closed = np.allclose(points[0], points[-1], atol=1e-2)
    spline_points = points[:-1] if is_closed else points
    spline_normals = normals[:-1] if is_closed else normals

    if len(spline_points) < 2:
        raise ValueError("not enough points for spline")

    periodic = is_closed and len(spline_points) >= 4
    degree = min(3, len(spline_points) - 1)
    u_fine = np.linspace(0, 1, POINTS_PER_LAYER)

    tck_points, _ = splprep(spline_points.T, s=smoothing, per=periodic, k=degree)
    smooth_points = np.array(splev(u_fine, tck_points)).T

    tck_normals, _ = splprep(spline_normals.T, s=smoothing, per=periodic, k=degree)
    smooth_normals = _normalize_rows(np.array(splev(u_fine, tck_normals)).T)

    if periodic and not np.allclose(smooth_points[0], smooth_points[-1]):
        smooth_points = np.vstack([smooth_points, smooth_points[0]])
        smooth_normals = np.vstack([smooth_normals, smooth_normals[0]])

    return smooth_points, smooth_normals


def generate_layered_path(mesh: trimesh.Trimesh, num_layers: int = 30, smoothing: float = 0.1) -> LayeredPathData:
    mesh.visual.face_colors = np.full((len(mesh.faces), 4), [180, 180, 180, 255], dtype=np.uint8)

    bounds = mesh.bounds
    model_scale = np.max(mesh.extents)
    z_levels = np.linspace(bounds[0][2], bounds[1][2], num_layers)

    data = LayeredPathData([], [], [], [])
    last_layer_endpoint = None
    forward = True

    print(f"Slicing model from Z={bounds[0][2]:.2f} to Z={bounds[1][2]:.2f}")

    for z in z_levels:
        try:
            section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        except Exception:
            continue

        if section is None or len(section.entities) == 0:
            continue

        layer_points = _outer_loop_points(section)
        if layer_points is None or len(layer_points) < 2:
            continue

        layer_normals = _raycast_normals(mesh, layer_points, model_scale, z)
        ordered_points, ordered_normals, last_layer_endpoint = _ordered_layer(
            layer_points, layer_normals, last_layer_endpoint, forward
        )
        forward = not forward

        try:
            smooth_points, smooth_normals = _smooth_layer(ordered_points, ordered_normals, smoothing)
            data.smooth_paths.append(smooth_points)
            data.smooth_normals.append(smooth_normals)
        except Exception as exc:
            print(f"Spline error at Z={z:.2f}: {exc}. Using raw points.")
            data.fallback_paths.append(ordered_points)
            data.fallback_normals.append(ordered_normals)

    return data


class MeshVisualizer:
    def __init__(
        self,
        mesh_path: str,
        wcs_origin: np.ndarray,
        stl_target_position: np.ndarray,
        num_layers: int = 150,
        axis_length: float = 10.0,
    ):
        self.mesh_path = mesh_path
        self.wcs_origin = np.asarray(wcs_origin, dtype=float)
        self.stl_target_position = np.asarray(stl_target_position, dtype=float)
        self.num_layers = num_layers
        self.axis_length = axis_length

        self.mesh: trimesh.Trimesh | None = None
        self.points = np.empty((0, 3))
        self.normals = np.empty((0, 3))
        self.smooth_paths: List[np.ndarray] = []
        self.fallback_paths: List[np.ndarray] = []
        self.toolpath_data: List[Dict[str, float]] = []
        self.frame_meshes: List[Tuple[pv.PolyData, str, int]] = []

        self._load_and_position_mesh()

    def _load_and_position_mesh(self) -> None:
        mesh = trimesh.load(self.mesh_path, force="mesh")
        mesh.apply_translation(self.stl_target_position - mesh.centroid)
        self.mesh = mesh
        print(f"Loaded mesh. Centroid: {self.mesh.centroid}")

    def generate_path_data(self, smoothing: float = 0.1) -> None:
        if self.mesh is None:
            raise ValueError("Mesh has not been loaded.")

        path_data = generate_layered_path(self.mesh, self.num_layers, smoothing)
        self.smooth_paths = path_data.smooth_paths
        self.fallback_paths = path_data.fallback_paths
        self.points = path_data.points
        self.normals = path_data.normals
        print(f"Generated {len(self.points)} points and {len(self.normals)} normals.")

    def generate_frames(self) -> None:
        self.frame_meshes = _axis_meshes(self.wcs_origin, self.axis_length)
        self.frame_meshes.extend(_axis_meshes(self.stl_target_position, self.axis_length * 0.5))

        if len(self.points) == 0 or len(self.normals) == 0:
            return

        frame_count = min(MAX_FRAME_PREVIEW_COUNT, len(self.points))
        sample_indices = np.linspace(0, len(self.points) - 1, frame_count, dtype=int)
        self.frame_meshes.extend(
            _sample_frame_meshes(
                self.points[sample_indices],
                self.normals[sample_indices],
                self.axis_length * 0.2,
            )
        )

    def generate_6dof_data(self) -> None:
        if len(self.points) != len(self.normals) or len(self.points) == 0:
            print("Warning: no matching points and normals for 6-DOF data.")
            return

        self.toolpath_data = []
        for point, normal in zip(self.points, self.normals):
            x, y, z = point - self.wcs_origin
            roll, pitch, yaw = _normal_to_rpy(normal)
            self.toolpath_data.append({"X": x, "Y": y, "Z": z, "A": roll, "B": pitch, "C": yaw})

        print(f"Generated {len(self.toolpath_data)} 6-DOF toolpath entries.")

    def print_6dof_data_for_index(self, index: int) -> None:
        if not self.toolpath_data:
            print("No 6-DOF toolpath data available.")
            return
        if not 0 <= index < len(self.toolpath_data):
            print(f"Index {index} is out of range 0..{len(self.toolpath_data) - 1}.")
            return

        data = self.toolpath_data[index]
        print(f"Point {index}:")
        print(f"XYZ: X={data['X']:.3f}, Y={data['Y']:.3f}, Z={data['Z']:.3f}")
        print(f"ABC: A={data['A']:.2f} deg, B={data['B']:.2f} deg, C={data['C']:.2f} deg")

    def get_scene(self) -> pv.Plotter:
        if self.mesh is None:
            raise ValueError("Mesh has not been loaded.")

        plotter = pv.Plotter()
        plotter.add_mesh(_trimesh_to_pyvista(self.mesh), color="lightgray", show_edges=False)

        _add_line_meshes(plotter, self.frame_meshes)
        self._add_toolpaths(plotter)
        self._add_pickable_points(plotter)
        return plotter

    def _add_toolpaths(self, plotter: pv.Plotter) -> None:
        for path in self.smooth_paths:
            if len(path) > 1:
                plotter.add_lines(path, color="green", width=2, connected=True)
        for path in self.fallback_paths:
            if len(path) > 1:
                plotter.add_lines(path, color="red", width=2, connected=True)

    def _add_pickable_points(self, plotter: pv.Plotter) -> None:
        if len(self.points) == 0 or not self.toolpath_data:
            return

        point_cloud = pv.PolyData(self.points)
        for key in self.toolpath_data[0]:
            point_cloud.point_data[key] = np.array([row[key] for row in self.toolpath_data])

        plotter.add_mesh(
            point_cloud,
            color="black",
            point_size=10,
            render_points_as_spheres=True,
            name="toolpath_points",
            pickable=True,
        )

        def show_picked_point(_picked_point=None) -> None:
            if plotter.actors.get("picked_info_text"):
                plotter.remove_actor("picked_info_text", render=False)
            if plotter.picked_point is None:
                return

            point_id = point_cloud.find_closest_point(plotter.picked_point)
            row = {key: point_cloud.point_data[key][point_id] for key in ("X", "Y", "Z", "A", "B", "C")}
            plotter.add_text(
                (
                    f"XYZ: ({row['X']:.3f}, {row['Y']:.3f}, {row['Z']:.3f})\n"
                    f"ABC: ({row['A']:.2f}, {row['B']:.2f}, {row['C']:.2f}) deg"
                ),
                position="lower_right",
                font_size=10,
                color="white",
                name="picked_info_text",
            )

        plotter.enable_point_picking(
            callback=show_picked_point,
            show_point=True,
            color="yellow",
            point_size=1,
        )
        print("Interactive point picking enabled.")


def main() -> None:
    visualizer = MeshVisualizer(
        mesh_path=r"C:\Users\shish\C-\Slicing\15778_NoveltyBust_EgyptianPharaoh_V1_NEW.obj",
        wcs_origin=np.array([0, 0, 0]),
        stl_target_position=np.array([10, 10, 10]),
        num_layers=10,
        axis_length=1,
    )

    visualizer.generate_path_data()
    visualizer.generate_frames()
    visualizer.generate_6dof_data()

    if visualizer.toolpath_data:
        visualizer.print_6dof_data_for_index(len(visualizer.toolpath_data) // 2)

    visualizer.get_scene().show()


if __name__ == "__main__":
    main()
