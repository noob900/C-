from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p
import pybullet_data
import pyvista as pv
from scipy.interpolate import splev, splprep
import trimesh
import vtk


EPSILON = 1e-9
POINTS_PER_LAYER = 400
MAX_FRAME_PREVIEW_COUNT = 20000


@dataclass
class LayeredPathData:
    path_points: List[np.ndarray]
    path_normals: List[np.ndarray]
    path_tangents: List[np.ndarray]
    smooth_paths: List[np.ndarray]
    fallback_paths: List[np.ndarray]
    smooth_normals: List[np.ndarray]
    fallback_normals: List[np.ndarray]
    smooth_tangents: List[np.ndarray]
    fallback_tangents: List[np.ndarray]

    @property
    def points(self) -> np.ndarray:
        return _stack_or_empty(self.path_points)

    @property
    def normals(self) -> np.ndarray:
        return _stack_or_empty(self.path_normals)

    @property
    def tangents(self) -> np.ndarray:
        return _stack_or_empty(self.path_tangents)


@dataclass
class ToolFrame:
    point: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray


@dataclass
class RobotFrame:
    tcp: ToolFrame
    flange: ToolFrame


@dataclass
class RobotVisualSettings:
    link_color: str = "orange"
    joint_color: str = "dodgerblue"
    base_color: str = "black"
    error_color: str = "tomato"
    joint_radius: float = 18.0
    link_radius: float = 6.0
    base_radius: float = 36.0
    base_height: float = 8.0


@dataclass
class SixAxisRobotConfig:
    urdf_path: str = "kuka_iiwa/model.urdf"
    end_effector_link_name: str = "lbr_iiwa_link_7"
    active_joint_count: int | None = None
    base_xyzabc: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    home_joint_angles_deg: np.ndarray | None = None
    scene_units_per_urdf_unit: float = 1000.0
    visuals: RobotVisualSettings = field(default_factory=RobotVisualSettings)


@dataclass
class RotaryTableConfig:
    enabled: bool = False
    center_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    axis_xyz: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    angle_deg: float = 0.0
    radius: float = 140.0
    height: float = 35.0
    color: str = "dimgray"


@dataclass
class SixAxisRobotPose:
    joint_angles: np.ndarray
    joint_points: np.ndarray
    transforms: List[np.ndarray]
    position_error: float
    rotation_error_deg: float
    success: bool


@dataclass
class RobotVisualMesh:
    link_index: int
    mesh: pv.PolyData
    local_transform: np.ndarray
    color: Tuple[float, float, float]
    opacity: float


def _stack_or_empty(parts: List[np.ndarray]) -> np.ndarray:
    valid_parts = [part for part in parts if len(part) > 0]
    return np.vstack(valid_parts) if valid_parts else np.empty((0, 3))


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + EPSILON)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPSILON)


def _fallback_tangent(z_axis: np.ndarray) -> np.ndarray:
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, z_axis)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    return _normalize(ref - z_axis * np.dot(ref, z_axis))


def _tool_frame_axes(normal: np.ndarray, tangent: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = _normalize(normal)

    if tangent is None or np.linalg.norm(tangent) < EPSILON:
        x_axis = _fallback_tangent(z_axis)
    else:
        x_axis = tangent - z_axis * np.dot(tangent, z_axis)
        if np.linalg.norm(x_axis) < EPSILON:
            x_axis = _fallback_tangent(z_axis)
        else:
            x_axis = _normalize(x_axis)

    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    return x_axis, y_axis, z_axis


def _axes_to_rpy(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> Tuple[float, float, float]:
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


def _rpy_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ]
    )
    rotation_y = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ]
    )
    rotation_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return rotation_z @ rotation_y @ rotation_x


def _axis_angle_to_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = _normalize(np.asarray(axis, dtype=float))
    angle = math.radians(angle_deg)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


def _matrix_from_axes(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    return np.array([x_axis, y_axis, z_axis], dtype=float).T


def _transform_from_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def _transform_from_xyzabc(xyzabc: np.ndarray) -> np.ndarray:
    x, y, z, a, b, c = np.asarray(xyzabc, dtype=float)
    return _transform_from_pose(np.array([x, y, z]), _rpy_to_matrix(a, b, c))


def _rotation_about_point_transform(center: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    rotation = _axis_angle_to_matrix(axis, angle_deg)
    center = np.asarray(center, dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center - rotation @ center
    return transform


def _modified_dh_transform(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _rotation_error_vector(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    error_rotation = target @ current.T
    trace = np.trace(error_rotation)
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(cos_angle)

    if angle < EPSILON:
        return np.zeros(3)

    if abs(math.sin(angle)) < EPSILON:
        return np.array(
            [
                error_rotation[2, 1] - error_rotation[1, 2],
                error_rotation[0, 2] - error_rotation[2, 0],
                error_rotation[1, 0] - error_rotation[0, 1],
            ]
        )

    axis = np.array(
        [
            error_rotation[2, 1] - error_rotation[1, 2],
            error_rotation[0, 2] - error_rotation[2, 0],
            error_rotation[1, 0] - error_rotation[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return axis * angle


def _quaternion_from_matrix(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif diagonal_index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale

    quaternion = np.array([qx, qy, qz, qw], dtype=float)
    quaternion /= np.linalg.norm(quaternion) + EPSILON
    return tuple(float(value) for value in quaternion)


def _matrix_from_quaternion(quaternion: Tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(p.getMatrixFromQuaternion(quaternion), dtype=float).reshape(3, 3)


def _transform_from_position_quaternion(position: np.ndarray, quaternion: Tuple[float, float, float, float]) -> np.ndarray:
    return _transform_from_pose(np.asarray(position, dtype=float), _matrix_from_quaternion(quaternion))


def _vtk_matrix_from_numpy(transform: np.ndarray) -> vtk.vtkMatrix4x4:
    matrix = vtk.vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            matrix.SetElement(row, column, float(transform[row, column]))
    return matrix


def _frame_to_transform(frame: ToolFrame) -> np.ndarray:
    return _transform_from_pose(frame.point, _matrix_from_axes(frame.x_axis, frame.y_axis, frame.z_axis))


def _tool_frame_from_transform(transform: np.ndarray, normal=None, tangent=None) -> ToolFrame:
    rotation = transform[:3, :3]
    return ToolFrame(
        point=transform[:3, 3].copy(),
        normal=_normalize(np.asarray(normal if normal is not None else rotation[:, 2], dtype=float)),
        tangent=_normalize(np.asarray(tangent if tangent is not None else rotation[:, 0], dtype=float)),
        x_axis=_normalize(rotation[:, 0]),
        y_axis=_normalize(rotation[:, 1]),
        z_axis=_normalize(rotation[:, 2]),
    )


def _axes_to_axis_angle(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> Tuple[float, float, float]:
    rotation = np.array([x_axis, y_axis, z_axis]).T
    trace = np.trace(rotation)
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(cos_angle)

    if abs(angle) < EPSILON:
        return 0.0, 0.0, 0.0

    sin_angle = math.sin(angle)
    if abs(sin_angle) < EPSILON:
        axis = np.array(
            [
                math.sqrt(max(0.0, (rotation[0, 0] + 1.0) * 0.5)),
                math.sqrt(max(0.0, (rotation[1, 1] + 1.0) * 0.5)),
                math.sqrt(max(0.0, (rotation[2, 2] + 1.0) * 0.5)),
            ]
        )
        axis[0] = math.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
        axis[1] = math.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
        axis[2] = math.copysign(axis[2], rotation[1, 0] - rotation[0, 1])
        axis = _normalize(axis)
    else:
        axis = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        ) / (2.0 * sin_angle)

    rx, ry, rz = axis * angle
    return rx, ry, rz


def _build_continuous_frames(
    points: np.ndarray,
    normals: np.ndarray,
    tangents: np.ndarray,
) -> List[ToolFrame]:
    frames: List[ToolFrame] = []

    for point, normal, tangent in zip(points, normals, tangents):
        x_axis, y_axis, z_axis = _tool_frame_axes(normal, tangent)

        if frames:
            previous = frames[-1]
            if np.dot(z_axis, previous.z_axis) < 0.0:
                z_axis = -z_axis
                normal = -normal

            x_axis = x_axis - z_axis * np.dot(x_axis, z_axis)
            if np.linalg.norm(x_axis) < EPSILON:
                x_axis = previous.x_axis - z_axis * np.dot(previous.x_axis, z_axis)

            if np.linalg.norm(x_axis) < EPSILON:
                x_axis = _fallback_tangent(z_axis)
            else:
                x_axis = _normalize(x_axis)

            if np.dot(x_axis, previous.x_axis) < 0.0:
                x_axis = -x_axis

            y_axis = _normalize(np.cross(z_axis, x_axis))
            x_axis = _normalize(np.cross(y_axis, z_axis))

        frames.append(
            ToolFrame(
                point=np.asarray(point, dtype=float),
                normal=np.asarray(normal, dtype=float),
                tangent=np.asarray(tangent, dtype=float),
                x_axis=x_axis,
                y_axis=y_axis,
                z_axis=z_axis,
            )
        )

    return frames


def _build_robot_frames(tool_frames: List[ToolFrame], tcp_offset_xyzabc: np.ndarray) -> List[RobotFrame]:
    flange_to_tcp = _transform_from_xyzabc(tcp_offset_xyzabc)
    tcp_to_flange = np.linalg.inv(flange_to_tcp)
    robot_frames: List[RobotFrame] = []

    for tcp_frame in tool_frames:
        tcp_transform = _frame_to_transform(tcp_frame)
        flange_transform = tcp_transform @ tcp_to_flange
        flange_frame = _tool_frame_from_transform(flange_transform)
        robot_frames.append(RobotFrame(tcp=tcp_frame, flange=flange_frame))

    return robot_frames


def _frame_normal_alignment(frames: List[ToolFrame], normals: np.ndarray) -> Tuple[float, float]:
    if not frames or len(normals) == 0:
        return 0.0, 0.0

    count = min(len(frames), len(normals))
    dots = np.array(
        [
            float(np.dot(frames[index].z_axis, _normalize(normals[index])))
            for index in range(count)
        ],
        dtype=float,
    )
    return float(np.min(dots)), float(np.mean(dots))


class SixAxisRobot:
    def __init__(self, config: SixAxisRobotConfig, wcs_origin: np.ndarray):
        self.config = config
        self.base_transform = _transform_from_xyzabc(config.base_xyzabc.copy())
        self.base_transform[:3, 3] += np.asarray(wcs_origin, dtype=float)
        self.scene_units_per_urdf_unit = float(config.scene_units_per_urdf_unit)
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        base_quaternion = _quaternion_from_matrix(self.base_transform[:3, :3])
        self.robot_id = p.loadURDF(
            config.urdf_path,
            basePosition=(self.base_transform[:3, 3] / self.scene_units_per_urdf_unit).tolist(),
            baseOrientation=base_quaternion,
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        self.joint_indices: List[int] = []
        self.joint_names: List[str] = []
        lower_limits = []
        upper_limits = []
        rest_poses = []
        self.end_effector_link_index = self._resolve_end_effector_link_index(config.end_effector_link_name)
        active_joint_count = config.active_joint_count

        for joint_index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            joint_info = p.getJointInfo(self.robot_id, joint_index, physicsClientId=self.client_id)
            joint_type = joint_info[2]
            if joint_type not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                continue
            if active_joint_count is not None and len(self.joint_indices) >= active_joint_count:
                continue

            lower_limit = float(joint_info[8])
            upper_limit = float(joint_info[9])
            if lower_limit >= upper_limit:
                lower_limit, upper_limit = -math.pi, math.pi

            self.joint_indices.append(joint_index)
            self.joint_names.append(joint_info[1].decode("utf-8"))
            lower_limits.append(lower_limit)
            upper_limits.append(upper_limit)
            rest_poses.append(float(np.clip(0.0, lower_limit, upper_limit)))

        if not self.joint_indices:
            raise ValueError(f"URDF '{config.urdf_path}' has no movable joints.")

        self.lower_limits = np.asarray(lower_limits, dtype=float)
        self.upper_limits = np.asarray(upper_limits, dtype=float)
        self.joint_ranges = self.upper_limits - self.lower_limits
        self.home_angles = np.asarray(rest_poses, dtype=float)
        if config.home_joint_angles_deg is not None:
            configured_home_deg = np.asarray(config.home_joint_angles_deg, dtype=float).reshape(-1)
            configured_home = np.radians(configured_home_deg)
            if configured_home.shape != self.home_angles.shape:
                raise ValueError(
                    f"home_joint_angles_deg must contain {len(self.home_angles)} values for this URDF; "
                    f"got {len(configured_home_deg)}. Joint order: {', '.join(self.joint_names)}."
                )
            self.home_angles = np.clip(configured_home, self.lower_limits, self.upper_limits)
        self._apply_joint_positions(self.home_angles)
        self.visual_meshes = self._load_visual_meshes()
        print(
            f"Loaded URDF robot '{config.urdf_path}' with {len(self.joint_indices)} movable joints. "
            f"End effector link: {self.end_effector_link_index}. "
            f"Visual meshes: {len(self.visual_meshes)}."
        )

    @staticmethod
    def default_config() -> SixAxisRobotConfig:
        return SixAxisRobotConfig()

    @property
    def joint_limits_deg(self) -> np.ndarray:
        return np.degrees(np.column_stack((self.lower_limits, self.upper_limits)))

    def _resolve_end_effector_link_index(self, link_name: str) -> int:
        fallback_index = p.getNumJoints(self.robot_id, physicsClientId=self.client_id) - 1
        for joint_index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            joint_info = p.getJointInfo(self.robot_id, joint_index, physicsClientId=self.client_id)
            if joint_info[12].decode("utf-8") == link_name:
                return joint_index
        print(f"Warning: link '{link_name}' was not found in URDF; using last link index {fallback_index}.")
        return fallback_index

    def _apply_joint_positions(self, joint_angles: np.ndarray) -> None:
        for joint_index, joint_angle in zip(self.joint_indices, joint_angles):
            p.resetJointState(
                self.robot_id,
                joint_index,
                float(joint_angle),
                physicsClientId=self.client_id,
            )

    def _link_transform(self, link_index: int) -> np.ndarray:
        link_state = p.getLinkState(
            self.robot_id,
            link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        position = np.asarray(link_state[4], dtype=float)
        position *= self.scene_units_per_urdf_unit
        rotation = np.asarray(p.getMatrixFromQuaternion(link_state[5]), dtype=float).reshape(3, 3)
        return _transform_from_pose(position, rotation)

    def _load_visual_meshes(self) -> List[RobotVisualMesh]:
        visual_meshes: List[RobotVisualMesh] = []
        for visual_index, visual_shape in enumerate(
            p.getVisualShapeData(self.robot_id, physicsClientId=self.client_id)
        ):
            link_index = int(visual_shape[1])
            geometry_type = int(visual_shape[2])
            if geometry_type != p.GEOM_MESH:
                continue

            mesh_path = visual_shape[4].decode("utf-8")
            if not mesh_path:
                continue

            try:
                mesh = pv.read(mesh_path)
            except Exception as exc:
                print(f"Warning: could not load URDF visual mesh '{mesh_path}': {exc}")
                continue

            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_geometry()

            mesh = mesh.copy(deep=True)
            mesh.points *= np.asarray(visual_shape[3], dtype=float) * self.scene_units_per_urdf_unit
            local_position = np.asarray(visual_shape[5], dtype=float) * self.scene_units_per_urdf_unit
            local_quaternion = tuple(float(value) for value in visual_shape[6])
            rgba = tuple(float(value) for value in visual_shape[7])
            visual_meshes.append(
                RobotVisualMesh(
                    link_index=link_index,
                    mesh=mesh,
                    local_transform=_transform_from_position_quaternion(local_position, local_quaternion),
                    color=rgba[:3],
                    opacity=rgba[3],
                )
            )

        if not visual_meshes:
            print("Warning: URDF has no readable visual meshes; using simple robot skeleton.")
        return visual_meshes

    def visual_mesh_instances(self, pose: SixAxisRobotPose) -> List[Tuple[pv.PolyData, Tuple[float, float, float], float]]:
        link_transforms = {-1: pose.transforms[0]}
        for transform_index, joint_index in enumerate(self.joint_indices, start=1):
            if transform_index < len(pose.transforms):
                link_transforms[joint_index] = pose.transforms[transform_index]

        instances = []
        for visual_mesh in self.visual_meshes:
            link_transform = link_transforms.get(visual_mesh.link_index)
            if link_transform is None:
                link_transform = self._link_transform(visual_mesh.link_index)

            mesh = visual_mesh.mesh.copy(deep=True)
            mesh.transform(link_transform @ visual_mesh.local_transform, inplace=True)
            instances.append((mesh, visual_mesh.color, visual_mesh.opacity))
        return instances

    def visual_actor_transforms(self, pose: SixAxisRobotPose) -> List[np.ndarray]:
        link_transforms = {-1: pose.transforms[0]}
        for transform_index, joint_index in enumerate(self.joint_indices, start=1):
            if transform_index < len(pose.transforms):
                link_transforms[joint_index] = pose.transforms[transform_index]

        transforms = []
        for visual_mesh in self.visual_meshes:
            link_transform = link_transforms.get(visual_mesh.link_index)
            if link_transform is None:
                link_transform = self._link_transform(visual_mesh.link_index)
            transforms.append(link_transform @ visual_mesh.local_transform)
        return transforms

    def forward_kinematics(self, joint_angles: np.ndarray) -> List[np.ndarray]:
        joint_angles = np.clip(np.asarray(joint_angles, dtype=float), self.lower_limits, self.upper_limits)
        self._apply_joint_positions(joint_angles)
        transforms = [self.base_transform.copy()]
        for joint_index in self.joint_indices:
            transforms.append(self._link_transform(joint_index))
        return transforms

    def _ik_seed_candidates(self, seed: np.ndarray) -> List[np.ndarray]:
        midpoint = (self.lower_limits + self.upper_limits) * 0.5
        span = self.upper_limits - self.lower_limits
        candidates = [
            seed,
            self.home_angles,
            midpoint,
        ]
        for scale in (-0.35, 0.35):
            candidates.append(midpoint + span * scale)

        alternating = np.ones(len(self.joint_indices))
        alternating[1::2] = -1.0
        candidates.append(midpoint + span * 0.25 * alternating)
        candidates.append(midpoint - span * 0.25 * alternating)

        unique_candidates = []
        for candidate in candidates:
            clipped = np.clip(np.asarray(candidate, dtype=float), self.lower_limits, self.upper_limits)
            if not any(np.allclose(clipped, existing, atol=1e-5) for existing in unique_candidates):
                unique_candidates.append(clipped)
        return unique_candidates

    def solve_ik(self, target_transform: np.ndarray, seed_angles: np.ndarray | None = None) -> SixAxisRobotPose:
        seed = self.home_angles if seed_angles is None else np.asarray(seed_angles, dtype=float)
        seed = np.clip(seed, self.lower_limits, self.upper_limits)
        target_position = (target_transform[:3, 3] / self.scene_units_per_urdf_unit).tolist()
        target_orientation = _quaternion_from_matrix(target_transform[:3, :3])
        best_pose = None
        best_score = float("inf")

        for rest_pose in self._ik_seed_candidates(seed):
            self._apply_joint_positions(rest_pose)
            result = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link_index,
                targetPosition=target_position,
                targetOrientation=target_orientation,
                lowerLimits=self.lower_limits.tolist(),
                upperLimits=self.upper_limits.tolist(),
                jointRanges=self.joint_ranges.tolist(),
                restPoses=rest_pose.tolist(),
                maxNumIterations=300,
                residualThreshold=1e-5,
                physicsClientId=self.client_id,
            )
            joint_angles = np.clip(
                np.asarray(result[: len(self.joint_indices)], dtype=float),
                self.lower_limits,
                self.upper_limits,
            )
            transforms = self.forward_kinematics(joint_angles)
            final_transform = transforms[-1]
            position_error = float(np.linalg.norm(target_transform[:3, 3] - final_transform[:3, 3]))
            rotation_error = float(
                np.linalg.norm(_rotation_error_vector(final_transform[:3, :3], target_transform[:3, :3]))
            )
            rotation_error_deg = math.degrees(rotation_error)
            pose = SixAxisRobotPose(
                joint_angles=joint_angles,
                joint_points=np.array([transform[:3, 3] for transform in transforms]),
                transforms=transforms,
                position_error=position_error,
                rotation_error_deg=rotation_error_deg,
                success=bool(position_error < 5.0 and rotation_error_deg < 10.0),
            )
            score = position_error + rotation_error_deg * 2.0
            if score < best_score:
                best_score = score
                best_pose = pose
            if pose.success:
                return pose

        return best_pose


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


def _axis_meshes(origin: np.ndarray, length: float, axes=None, colors=None) -> List[Tuple[pv.PolyData, str, int]]:
    if axes is None:
        axes = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

    if colors is None:
        colors = ["red", "green", "blue"]

    return [
        (_line_polydata([segment]), color, 3)
        for segment, color in zip(_axis_segments(origin, axes, length), colors)
    ]


def _sample_frame_meshes(
    frames: List[ToolFrame],
    length: float,
    colors=None,
    width: int = 1,
) -> List[Tuple[pv.PolyData, str, int]]:
    x_segments, y_segments, z_segments = [], [], []
    for frame in frames:
        x_segments.append(np.array([frame.point, frame.point + frame.x_axis * length]))
        y_segments.append(np.array([frame.point, frame.point + frame.y_axis * length]))
        z_segments.append(np.array([frame.point, frame.point + frame.z_axis * length]))

    if colors is None:
        colors = ["red", "green", "blue"]

    return [
        (_line_polydata(x_segments), colors[0], width),
        (_line_polydata(y_segments), colors[1], width),
        (_line_polydata(z_segments), colors[2], width),
    ]


def _add_line_meshes(plotter: pv.Plotter, meshes: List[Tuple[pv.PolyData, str, int]]) -> None:
    for mesh, color, width in meshes:
        if mesh.n_points > 0:
            plotter.add_mesh(mesh, color=color, line_width=width)


def _frame_axis_polydata(frame: ToolFrame, length: float) -> Tuple[pv.PolyData, pv.PolyData, pv.PolyData]:
    return tuple(
        _line_polydata([np.array([frame.point, frame.point + axis * length])])
        for axis in (frame.x_axis, frame.y_axis, frame.z_axis)
    )


def _outer_loop_points(section) -> np.ndarray | None:
    planar_path, to_3d = section.to_2D()
    loops = [np.asarray(loop) for loop in planar_path.discrete if len(loop) >= 3]
    if not loops:
        return None

    centroid = np.mean(np.vstack(loops), axis=0)
    outer_loop = max(loops, key=lambda loop: np.mean(np.linalg.norm(loop - centroid, axis=1)))
    if _signed_area_xy(outer_loop) < 0.0:
        outer_loop = np.flip(outer_loop, axis=0)

    z_values = np.zeros(len(outer_loop))
    return trimesh.transform_points(np.column_stack((outer_loop, z_values)), to_3d)


def _signed_area_xy(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0

    xy = np.asarray(points, dtype=float)[:, :2]
    return 0.5 * np.sum(
        xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]
    )


def _ensure_counter_clockwise(points: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if _signed_area_xy(points) >= 0.0:
        return points, normals
    return np.flip(points, axis=0), np.flip(normals, axis=0)


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

    locations, ray_indices, triangle_indices = mesh.ray.intersects_location(origins, directions)
    if len(ray_indices) == 0:
        print(f"Warning: normal raycast failed at Z={z:.2f}; using radial fallback.")
        return _fallback_normals(mesh, points)

    normals = np.zeros_like(points)
    hit_normals = mesh.face_normals[triangle_indices]
    hit_distances = np.linalg.norm(locations - origins[ray_indices], axis=1)
    for ray_index in np.unique(ray_indices):
        ray_hits = np.where(ray_indices == ray_index)[0]
        nearest_hit = ray_hits[np.argmin(hit_distances[ray_hits])]
        normal = hit_normals[nearest_hit]
        if np.dot(normal, directions[ray_index]) > 0.0:
            normal = -normal
        normals[ray_index] = normal

    return _fill_missing_normals(normals)


def _path_tangents(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3))
    if len(points) == 1:
        return np.array([_fallback_tangent(_normalize(normals[0]))])

    is_closed = np.allclose(points[0], points[-1], atol=1e-2)
    tangents = np.zeros_like(points)
    if is_closed and len(points) > 3:
        unique_points = points[:-1]
        unique_tangents = np.roll(unique_points, -1, axis=0) - np.roll(unique_points, 1, axis=0)
        tangents[:-1] = unique_tangents
        tangents[-1] = unique_tangents[0]
    else:
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        if len(points) > 2:
            tangents[1:-1] = points[2:] - points[:-2]

    projected = np.zeros_like(tangents)
    previous = None
    for index, (tangent, normal) in enumerate(zip(tangents, normals)):
        z_axis = _normalize(normal)
        x_axis = tangent - z_axis * np.dot(tangent, z_axis)
        if np.linalg.norm(x_axis) < EPSILON:
            x_axis = previous if previous is not None else _fallback_tangent(z_axis)
        else:
            x_axis = _normalize(x_axis)

        projected[index] = x_axis
        previous = x_axis

    return projected


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


def generate_layered_path(
    mesh: trimesh.Trimesh,
    num_layers: int = 30,
    smoothing: float = 0.1,
    toolpath_mode: str = "layers",
    top_section_offset: float = 0.5,
) -> LayeredPathData:
    mesh.visual.face_colors = np.full((len(mesh.faces), 4), [180, 180, 180, 255], dtype=np.uint8)

    bounds = mesh.bounds
    model_scale = np.max(mesh.extents)
    mode = toolpath_mode.lower().strip()
    if mode not in {"layers", "top"}:
        raise ValueError("toolpath_mode must be either 'layers' or 'top'.")

    if mode == "top":
        offset = max(float(top_section_offset), model_scale * 1e-5, EPSILON)
        z_levels = np.array([bounds[1][2] - offset], dtype=float)
    else:
        z_levels = np.linspace(bounds[0][2], bounds[1][2], num_layers)

    data = LayeredPathData([], [], [], [], [], [], [], [], [])
    last_layer_endpoint = None

    if mode == "top":
        print(f"Generating top-only toolpath at Z={z_levels[0]:.2f} (top Z={bounds[1][2]:.2f}).")
    else:
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

        if mode == "top":
            layer_normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(layer_points), 1))
        else:
            layer_normals = _raycast_normals(mesh, layer_points, model_scale, z)
        ordered_points, ordered_normals, last_layer_endpoint = _ordered_layer(
            layer_points, layer_normals, last_layer_endpoint, True
        )
        ordered_points, ordered_normals = _ensure_counter_clockwise(ordered_points, ordered_normals)

        try:
            smooth_points, smooth_normals = _smooth_layer(ordered_points, ordered_normals, smoothing)
            smooth_points, smooth_normals = _ensure_counter_clockwise(smooth_points, smooth_normals)
            if mode == "top":
                smooth_normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(smooth_points), 1))
            else:
                smooth_normals = _raycast_normals(mesh, smooth_points, model_scale, z)
            smooth_tangents = _path_tangents(smooth_points, smooth_normals)
            data.path_points.append(smooth_points)
            data.path_normals.append(smooth_normals)
            data.path_tangents.append(smooth_tangents)
            data.smooth_paths.append(smooth_points)
            data.smooth_normals.append(smooth_normals)
            data.smooth_tangents.append(smooth_tangents)
        except Exception as exc:
            print(f"Spline error at Z={z:.2f}: {exc}. Using raw points.")
            ordered_tangents = _path_tangents(ordered_points, ordered_normals)
            data.path_points.append(ordered_points)
            data.path_normals.append(ordered_normals)
            data.path_tangents.append(ordered_tangents)
            data.fallback_paths.append(ordered_points)
            data.fallback_normals.append(ordered_normals)
            data.fallback_tangents.append(ordered_tangents)

    return data


class MeshVisualizer:
    def __init__(
        self,
        mesh_path: str,
        wcs_origin: np.ndarray,
        stl_target_position: np.ndarray,
        stl_target_abc: np.ndarray | None = None,
        num_layers: int = 150,
        axis_length: float = 10.0,
        toolpath_clearance: float = 0.0,
        toolpath_mode: str = "layers",
        top_section_offset: float = 0.5,
        tcp_offset_xyzabc: np.ndarray | None = None,
        show_flange_frames: bool = False,
        show_robot_kinematics: bool = True,
        show_playback_buttons: bool = False,
        auto_start_playback: bool = True,
        robot_kinematics_config: SixAxisRobotConfig | None = None,
        rotary_table_config: RotaryTableConfig | None = None,
    ):
        self.mesh_path = mesh_path
        self.wcs_origin = np.asarray(wcs_origin, dtype=float)
        self.stl_target_position = np.asarray(stl_target_position, dtype=float)
        self.stl_target_abc = (
            np.zeros(3, dtype=float)
            if stl_target_abc is None
            else np.asarray(stl_target_abc, dtype=float)
        )
        if self.stl_target_abc.shape != (3,):
            raise ValueError("stl_target_abc must contain exactly three values: A, B, C.")
        self.stl_target_rotation = _rpy_to_matrix(*self.stl_target_abc)
        self.num_layers = num_layers
        self.axis_length = axis_length
        self.toolpath_clearance = float(toolpath_clearance)
        self.toolpath_mode = toolpath_mode
        self.top_section_offset = float(top_section_offset)
        self.show_flange_frames = show_flange_frames
        self.show_robot_kinematics = show_robot_kinematics
        self.show_playback_buttons = show_playback_buttons
        self.auto_start_playback = auto_start_playback
        self.rotary_table_config = rotary_table_config or RotaryTableConfig()
        self.rotary_table_config.center_xyz = np.asarray(self.rotary_table_config.center_xyz, dtype=float)
        self.rotary_table_config.axis_xyz = _normalize(np.asarray(self.rotary_table_config.axis_xyz, dtype=float))
        self.tcp_offset_xyzabc = (
            np.zeros(6, dtype=float)
            if tcp_offset_xyzabc is None
            else np.asarray(tcp_offset_xyzabc, dtype=float)
        )
        if self.tcp_offset_xyzabc.shape != (6,):
            raise ValueError("tcp_offset_xyzabc must contain exactly six values: X, Y, Z, A, B, C.")

        self.mesh: trimesh.Trimesh | None = None
        self.points = np.empty((0, 3))
        self.normals = np.empty((0, 3))
        self.tangents = np.empty((0, 3))
        self.tool_frames: List[ToolFrame] = []
        self.robot_frames: List[RobotFrame] = []
        self.smooth_paths: List[np.ndarray] = []
        self.fallback_paths: List[np.ndarray] = []
        self.toolpath_data: List[Dict[str, float]] = []
        self.frame_meshes: List[Tuple[pv.PolyData, str, int]] = []
        self.six_axis_robot = SixAxisRobot(
            robot_kinematics_config or SixAxisRobot.default_config(),
            self.wcs_origin,
        )
        self.last_joint_angles = self.six_axis_robot.home_angles.copy()

        self._load_and_position_mesh()

    def _load_and_position_mesh(self) -> None:
        mesh = trimesh.load(self.mesh_path, force="mesh")
        mesh.apply_translation(-mesh.centroid)
        mesh.apply_transform(_transform_from_pose(np.zeros(3), self.stl_target_rotation))
        mesh.apply_translation(self.stl_target_position)
        if self.rotary_table_config.enabled:
            mesh.apply_transform(
                _rotation_about_point_transform(
                    self.rotary_table_config.center_xyz,
                    self.rotary_table_config.axis_xyz,
                    self.rotary_table_config.angle_deg,
                )
            )
        self.mesh = mesh
        table_text = (
            f", table angle: {self.rotary_table_config.angle_deg:.2f} deg"
            if self.rotary_table_config.enabled
            else ""
        )
        print(f"Loaded mesh. Centroid: {self.mesh.centroid}, ABC: {self.stl_target_abc}{table_text}")

    def generate_path_data(self, smoothing: float = 0.1) -> None:
        if self.mesh is None:
            raise ValueError("Mesh has not been loaded.")

        path_data = generate_layered_path(
            self.mesh,
            self.num_layers,
            smoothing,
            self.toolpath_mode,
            self.top_section_offset,
        )
        self.smooth_paths = path_data.smooth_paths
        self.fallback_paths = path_data.fallback_paths
        self.normals = path_data.normals
        self.points = path_data.points + self.normals * self.toolpath_clearance
        self.tangents = path_data.tangents
        self.tool_frames = _build_continuous_frames(self.points, self.normals, self.tangents)
        self.robot_frames = _build_robot_frames(self.tool_frames, self.tcp_offset_xyzabc)
        min_alignment, mean_alignment = _frame_normal_alignment(self.tool_frames, self.normals)
        print(
            f"Generated {len(self.points)} points, {len(self.normals)} normals, "
            f"{len(self.tangents)} tangents, and {len(self.robot_frames)} robot frames. "
            f"Toolpath clearance: {self.toolpath_clearance:.2f} mm. "
            f"TCP/path Z alignment: min={min_alignment:.3f}, mean={mean_alignment:.3f}."
        )

    def generate_frames(self) -> None:
        self.frame_meshes = _axis_meshes(self.wcs_origin, self.axis_length)
        self.frame_meshes.extend(
            _axis_meshes(
                self.stl_target_position,
                self.axis_length * 0.5,
                axes=(
                    self.stl_target_rotation[:, 0],
                    self.stl_target_rotation[:, 1],
                    self.stl_target_rotation[:, 2],
                ),
            )
        )

        if not self.robot_frames:
            return

        frame_count = min(MAX_FRAME_PREVIEW_COUNT, len(self.robot_frames))
        sample_indices = np.linspace(0, len(self.robot_frames) - 1, frame_count, dtype=int)
        sampled_tcp_frames = [self.robot_frames[index].tcp for index in sample_indices]
        sampled_flange_frames = [self.robot_frames[index].flange for index in sample_indices]
        self.frame_meshes.extend(
            _sample_frame_meshes(
                sampled_tcp_frames,
                self.axis_length * 0.2,
                colors=["red", "green", "blue"],
            )
        )
        if self.show_flange_frames:
            self.frame_meshes.extend(
                _sample_frame_meshes(
                    sampled_flange_frames,
                    self.axis_length * 0.18,
                    colors=["magenta", "lime", "cyan"],
                )
            )

    def generate_6dof_data(self) -> None:
        if not self.robot_frames:
            print("Warning: no robot frames for 6-DOF data.")
            return

        self.toolpath_data = []
        for index, robot_frame in enumerate(self.robot_frames):
            tcp = robot_frame.tcp
            flange = robot_frame.flange
            tcp_x, tcp_y, tcp_z = tcp.point - self.wcs_origin
            flange_x, flange_y, flange_z = flange.point - self.wcs_origin
            tcp_roll, tcp_pitch, tcp_yaw = _axes_to_rpy(tcp.x_axis, tcp.y_axis, tcp.z_axis)
            flange_roll, flange_pitch, flange_yaw = _axes_to_rpy(
                flange.x_axis,
                flange.y_axis,
                flange.z_axis,
            )
            tcp_rx, tcp_ry, tcp_rz = _axes_to_axis_angle(tcp.x_axis, tcp.y_axis, tcp.z_axis)
            flange_rx, flange_ry, flange_rz = _axes_to_axis_angle(
                flange.x_axis,
                flange.y_axis,
                flange.z_axis,
            )
            self.toolpath_data.append(
                {
                    "index": index,
                    "X": tcp_x,
                    "Y": tcp_y,
                    "Z": tcp_z,
                    "A": tcp_roll,
                    "B": tcp_pitch,
                    "C": tcp_yaw,
                    "Rx": tcp_rx,
                    "Ry": tcp_ry,
                    "Rz": tcp_rz,
                    "tcp_x": tcp_x,
                    "tcp_y": tcp_y,
                    "tcp_z": tcp_z,
                    "tcp_a": tcp_roll,
                    "tcp_b": tcp_pitch,
                    "tcp_c": tcp_yaw,
                    "tcp_rx": tcp_rx,
                    "tcp_ry": tcp_ry,
                    "tcp_rz": tcp_rz,
                    "flange_x": flange_x,
                    "flange_y": flange_y,
                    "flange_z": flange_z,
                    "flange_a": flange_roll,
                    "flange_b": flange_pitch,
                    "flange_c": flange_yaw,
                    "flange_rx": flange_rx,
                    "flange_ry": flange_ry,
                    "flange_rz": flange_rz,
                    "tcp_offset_x": self.tcp_offset_xyzabc[0],
                    "tcp_offset_y": self.tcp_offset_xyzabc[1],
                    "tcp_offset_z": self.tcp_offset_xyzabc[2],
                    "tcp_offset_a": self.tcp_offset_xyzabc[3],
                    "tcp_offset_b": self.tcp_offset_xyzabc[4],
                    "tcp_offset_c": self.tcp_offset_xyzabc[5],
                    "x_axis_x": tcp.x_axis[0],
                    "x_axis_y": tcp.x_axis[1],
                    "x_axis_z": tcp.x_axis[2],
                    "y_axis_x": tcp.y_axis[0],
                    "y_axis_y": tcp.y_axis[1],
                    "y_axis_z": tcp.y_axis[2],
                    "z_axis_x": tcp.z_axis[0],
                    "z_axis_y": tcp.z_axis[1],
                    "z_axis_z": tcp.z_axis[2],
                }
            )

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
        self._add_rotary_table(plotter)
        plotter.add_mesh(_trimesh_to_pyvista(self.mesh), color="lightgray", show_edges=False)

        _add_line_meshes(plotter, self.frame_meshes)
        self._add_toolpaths(plotter)
        self._add_robot_playback(plotter)
        self._add_pickable_points(plotter)
        return plotter

    def _add_rotary_table(self, plotter: pv.Plotter) -> None:
        config = self.rotary_table_config
        if not config.enabled:
            return

        center = np.asarray(config.center_xyz, dtype=float)
        axis = _normalize(np.asarray(config.axis_xyz, dtype=float))
        radius = max(float(config.radius), EPSILON)
        height = max(float(config.height), EPSILON)
        body_center = center - axis * (height * 0.5)
        frame_length = max(radius * 0.45, self.axis_length * 2.0)
        angle_transform = _axis_angle_to_matrix(axis, config.angle_deg)

        plotter.add_mesh(
            pv.Cylinder(
                center=body_center,
                direction=axis,
                radius=radius,
                height=height,
                resolution=96,
            ),
            color=config.color,
            opacity=0.38,
            name="rotary_table_body",
        )

        axis_line = _line_polydata([np.array([center - axis * height, center + axis * radius])])
        plotter.add_mesh(axis_line, color="purple", line_width=5, name="rotary_table_axis")

        x_axis = angle_transform @ _fallback_tangent(axis)
        y_axis = _normalize(np.cross(axis, x_axis))
        table_frame = _axis_meshes(
            center,
            frame_length,
            axes=(x_axis, y_axis, axis),
            colors=("red", "green", "purple"),
        )
        _add_line_meshes(plotter, table_frame)

    def _add_toolpaths(self, plotter: pv.Plotter) -> None:
        for path in self.smooth_paths:
            if len(path) > 1:
                plotter.add_lines(path, color="darkgreen", width=1, connected=True)
        for path in self.fallback_paths:
            if len(path) > 1:
                plotter.add_lines(path, color="darkred", width=1, connected=True)
        if len(self.points) > 1 and abs(self.toolpath_clearance) > EPSILON:
            plotter.add_lines(self.points, color="cyan", width=3, connected=True)

    def _add_robot_playback(self, plotter: pv.Plotter) -> None:
        if not self.robot_frames:
            return

        axis_length = self.axis_length * 0.6
        marker_radius = max(self.axis_length * 0.08, EPSILON)
        target_fps = 60.0
        max_path_keyframes = 420
        interpolation_steps = 12
        stride = max(1, len(self.robot_frames) // max_path_keyframes)
        state = {
            "index": 0,
            "playing": False,
            "paused": False,
            "playback_generation": 0,
            "playback_cursor": 0,
            "speed": 1.0,
            "jog_angles_deg": np.degrees(self.last_joint_angles).copy(),
            "wcs_jog_xyzabc": np.zeros(6, dtype=float),
            "trajectory": None,
            "playback_frames": [],
        }
        joint_count = len(self.six_axis_robot.joint_indices)
        actor_names = [
            "animated_tcp_marker",
            "animated_flange_marker",
            "animated_tcp_x",
            "animated_tcp_y",
            "animated_tcp_z",
            "animated_tcp_flange_link",
            "animated_frame_text",
            "animated_pose_text",
            "home_pose_text",
            "machining_start_text",
            "machining_start_marker",
            "machining_start_flange_marker",
            "robot_jog_text",
            "wcs_jog_text",
            "six_axis_robot_base",
        ]
        actor_names.extend([f"six_axis_robot_joint_{index}" for index in range(joint_count + 1)])
        actor_names.extend([f"six_axis_robot_link_{index}" for index in range(joint_count)])
        actor_names.extend(
            [
                f"six_axis_robot_visual_{index}"
                for index in range(len(self.six_axis_robot.visual_meshes))
            ]
        )
        if self.show_flange_frames:
            actor_names.extend(["animated_flange_x", "animated_flange_y", "animated_flange_z"])

        def remove_animation_actors() -> None:
            for name in actor_names:
                if plotter.actors.get(name):
                    plotter.remove_actor(name, render=False)

        def add_momentary_button(callback, **widget_kwargs):
            widget_ref = {"widget": None, "resetting": False}

            def wrapped_callback(_state: bool) -> None:
                if widget_ref["resetting"]:
                    return

                callback()
                widget = widget_ref["widget"]
                if widget is not None:
                    widget_ref["resetting"] = True
                    widget.GetRepresentation().SetState(0)
                    widget_ref["resetting"] = False

            widget_ref["widget"] = plotter.add_checkbox_button_widget(
                wrapped_callback,
                value=False,
                **widget_kwargs,
            )
            return widget_ref["widget"]

        def add_frame_axes(frame: ToolFrame, prefix: str, colors: List[str], width: int) -> None:
            for suffix, mesh, color in zip(("x", "y", "z"), _frame_axis_polydata(frame, axis_length), colors):
                plotter.add_mesh(mesh, color=color, line_width=width, name=f"{prefix}_{suffix}")

        def format_joint_text(joint_angles_deg: np.ndarray) -> str:
            return "\n".join(
                f"J{index + 1}: {angle:.2f} deg"
                for index, angle in enumerate(joint_angles_deg)
            )

        def add_six_axis_robot_pose(pose: SixAxisRobotPose) -> None:
            visuals = self.six_axis_robot.config.visuals
            robot_color = visuals.link_color if pose.success else visuals.error_color
            joint_color = visuals.joint_color if pose.success else visuals.error_color
            joint_radius = max(visuals.joint_radius, EPSILON)
            link_radius = max(visuals.link_radius, EPSILON)

            if self.six_axis_robot.visual_meshes:
                add_or_update_visual_robot_pose(pose)
                return

            for link_index, (start, end) in enumerate(zip(pose.joint_points[:-1], pose.joint_points[1:])):
                direction = end - start
                length = float(np.linalg.norm(direction))
                if length < EPSILON:
                    continue
                plotter.add_mesh(
                    pv.Cylinder(
                        center=(start + end) * 0.5,
                        direction=direction,
                        radius=link_radius,
                        height=length,
                    ),
                    color=robot_color,
                    name=f"six_axis_robot_link_{link_index}",
                )

            plotter.add_mesh(
                pv.Cylinder(
                    center=pose.joint_points[0] - np.array([0.0, 0.0, visuals.base_height * 0.5]),
                    direction=(0.0, 0.0, 1.0),
                    radius=max(visuals.base_radius, EPSILON),
                    height=max(visuals.base_height, EPSILON),
                ),
                color=visuals.base_color,
                name="six_axis_robot_base",
            )
            for joint_index, point in enumerate(pose.joint_points):
                plotter.add_mesh(
                    pv.Sphere(radius=joint_radius, center=point),
                    color=joint_color,
                    name=f"six_axis_robot_joint_{joint_index}",
                )

        def add_or_update_visual_robot_pose(pose: SixAxisRobotPose) -> bool:
            if not self.six_axis_robot.visual_meshes:
                return False

            transforms = self.six_axis_robot.visual_actor_transforms(pose)
            existing_actors = [
                plotter.actors.get(f"six_axis_robot_visual_{mesh_index}")
                for mesh_index in range(len(self.six_axis_robot.visual_meshes))
            ]
            if all(actor is not None for actor in existing_actors):
                for actor, transform in zip(existing_actors, transforms):
                    actor.SetUserMatrix(_vtk_matrix_from_numpy(transform))
                return True

            for mesh_index, visual_mesh in enumerate(self.six_axis_robot.visual_meshes):
                mesh = visual_mesh.mesh.copy(deep=True)
                actor = plotter.add_mesh(
                    mesh,
                    color=visual_mesh.color,
                    opacity=visual_mesh.opacity,
                    smooth_shading=True,
                    name=f"six_axis_robot_visual_{mesh_index}",
                )
                actor.SetUserMatrix(_vtk_matrix_from_numpy(transforms[mesh_index]))
            return True

        def pose_from_joint_angles(joint_angles: np.ndarray) -> SixAxisRobotPose:
            transforms = self.six_axis_robot.forward_kinematics(joint_angles)
            return SixAxisRobotPose(
                joint_angles=joint_angles.copy(),
                joint_points=np.array([transform[:3, 3] for transform in transforms]),
                transforms=transforms,
                position_error=0.0,
                rotation_error_deg=0.0,
                success=True,
            )

        def flange_xyzabc_from_transform(transform: np.ndarray) -> np.ndarray:
            frame = _tool_frame_from_transform(transform)
            x, y, z = frame.point - self.wcs_origin
            a, b, c = _axes_to_rpy(frame.x_axis, frame.y_axis, frame.z_axis)
            return np.array([x, y, z, a, b, c], dtype=float)

        def sync_wcs_jog_from_current_robot() -> None:
            current_transform = self.six_axis_robot.forward_kinematics(self.last_joint_angles)[-1]
            state["wcs_jog_xyzabc"] = flange_xyzabc_from_transform(current_transform)

        def clear_lightweight_pose_actors() -> None:
            for name in (
                "animated_tcp_marker",
                "animated_flange_marker",
                "animated_tcp_x",
                "animated_tcp_y",
                "animated_tcp_z",
                "animated_tcp_flange_link",
                "animated_frame_text",
                "animated_pose_text",
                "home_pose_text",
                "machining_start_text",
                "machining_start_marker",
                "machining_start_flange_marker",
                "robot_jog_text",
                "wcs_jog_text",
            ):
                if plotter.actors.get(name):
                    plotter.remove_actor(name, render=False)

        def machining_start_pose(seed_angles: np.ndarray | None = None) -> SixAxisRobotPose:
            return self.six_axis_robot.solve_ik(
                _frame_to_transform(self.robot_frames[0].flange),
                self.six_axis_robot.home_angles if seed_angles is None else seed_angles,
            )

        def show_home_pose(_state: bool = False) -> None:
            if state["playing"]:
                return

            joint_angles = self.six_axis_robot.home_angles.copy()
            pose = pose_from_joint_angles(joint_angles)
            self.last_joint_angles = joint_angles.copy()
            state["jog_angles_deg"] = np.degrees(joint_angles).copy()
            state["wcs_jog_xyzabc"] = flange_xyzabc_from_transform(pose.transforms[-1])

            clear_lightweight_pose_actors()
            if not add_or_update_visual_robot_pose(pose):
                add_six_axis_robot_pose(pose)
            plotter.add_text(
                (
                    "Home position\n"
                    f"{format_joint_text(state['jog_angles_deg'])}"
                ),
                position="lower_right",
                font_size=10,
                color="black",
                name="home_pose_text",
            )
            plotter.render()

        def show_machining_start_pose(_state: bool = False) -> None:
            if state["playing"]:
                return

            pose = machining_start_pose(self.last_joint_angles)
            self.last_joint_angles = pose.joint_angles.copy()
            state["jog_angles_deg"] = np.degrees(pose.joint_angles).copy()
            state["wcs_jog_xyzabc"] = flange_xyzabc_from_transform(pose.transforms[-1])
            robot_frame = self.robot_frames[0]

            clear_lightweight_pose_actors()
            if not add_or_update_visual_robot_pose(pose):
                add_six_axis_robot_pose(pose)
            plotter.add_mesh(
                pv.Sphere(radius=marker_radius * 1.4, center=robot_frame.tcp.point),
                color="lime",
                name="machining_start_marker",
            )
            plotter.add_mesh(
                pv.Sphere(radius=marker_radius * 1.1, center=robot_frame.flange.point),
                color="white",
                name="machining_start_flange_marker",
            )
            add_frame_axes(robot_frame.tcp, "animated_tcp", ["red", "green", "blue"], 5)
            plotter.add_mesh(
                _line_polydata([np.array([robot_frame.flange.point, robot_frame.tcp.point])]),
                color="white",
                line_width=2,
                name="animated_tcp_flange_link",
            )
            plotter.add_text(
                (
                    "Machining start\n"
                    f"Frame 1/{len(self.robot_frames)}\n"
                    f"{format_joint_text(state['jog_angles_deg'])}\n"
                    f"IK pos err: {pose.position_error:.2f}\n"
                    f"IK rot err: {pose.rotation_error_deg:.2f} deg"
                ),
                position="lower_right",
                font_size=10,
                color="black",
                name="machining_start_text",
            )
            plotter.render()

        def show_joint_jog_pose() -> None:
            if state["playing"]:
                return

            joint_angles = np.radians(state["jog_angles_deg"])
            self.last_joint_angles = joint_angles.copy()
            pose = pose_from_joint_angles(joint_angles)
            state["wcs_jog_xyzabc"] = flange_xyzabc_from_transform(pose.transforms[-1])
            flange_x, flange_y, flange_z, flange_a, flange_b, flange_c = state["wcs_jog_xyzabc"]

            remove_animation_actors()
            add_six_axis_robot_pose(pose)
            plotter.add_text(
                (
                    "Robot jog\n"
                    f"{format_joint_text(state['jog_angles_deg'])}\n"
                    f"Flange XYZ: ({flange_x:.1f}, {flange_y:.1f}, {flange_z:.1f})\n"
                    f"Flange ABC: ({flange_a:.1f}, {flange_b:.1f}, {flange_c:.1f}) deg"
                ),
                position="lower_right",
                font_size=10,
                color="black",
                name="robot_jog_text",
            )
            plotter.render()

        def update_joint_jog(joint_index: int, value: float) -> None:
            state["jog_angles_deg"][joint_index] = float(value)
            show_joint_jog_pose()

        def show_wcs_jog_pose() -> None:
            if state["playing"]:
                return

            x, y, z, a, b, c = state["wcs_jog_xyzabc"]
            target_transform = _transform_from_pose(
                self.wcs_origin + np.array([x, y, z], dtype=float),
                _rpy_to_matrix(a, b, c),
            )
            pose = self.six_axis_robot.solve_ik(target_transform, self.last_joint_angles)
            self.last_joint_angles = pose.joint_angles.copy()
            state["jog_angles_deg"] = np.degrees(pose.joint_angles).copy()

            remove_animation_actors()
            add_six_axis_robot_pose(pose)
            plotter.add_text(
                (
                    "WCS jog\n"
                    f"X: {x:.1f} mm\n"
                    f"Y: {y:.1f} mm\n"
                    f"Z: {z:.1f} mm\n"
                    f"A: {a:.1f} deg\n"
                    f"B: {b:.1f} deg\n"
                    f"C: {c:.1f} deg\n"
                    f"{format_joint_text(state['jog_angles_deg'])}\n"
                    f"IK pos err: {pose.position_error:.2f}\n"
                    f"IK rot err: {pose.rotation_error_deg:.2f} deg"
                ),
                position="lower_right",
                font_size=10,
                color="black",
                name="wcs_jog_text",
            )
            plotter.render()

        def jog_wcs_axis(axis_index: int, step: float) -> None:
            state["wcs_jog_xyzabc"][axis_index] += step
            show_wcs_jog_pose()

        def show_robot_frame(index: int) -> None:
            state["index"] = int(np.clip(index, 0, len(self.robot_frames) - 1))
            index = state["index"]
            robot_frame = self.robot_frames[index]
            remove_animation_actors()
            tcp_data = self.toolpath_data[index] if self.toolpath_data else None
            six_axis_pose = None
            if self.show_robot_kinematics:
                six_axis_pose = self.six_axis_robot.solve_ik(
                    _frame_to_transform(robot_frame.flange),
                    self.last_joint_angles,
                )
                self.last_joint_angles = six_axis_pose.joint_angles.copy()
                state["jog_angles_deg"] = np.degrees(six_axis_pose.joint_angles).copy()
                state["wcs_jog_xyzabc"] = flange_xyzabc_from_transform(six_axis_pose.transforms[-1])
                add_six_axis_robot_pose(six_axis_pose)

            plotter.add_mesh(
                pv.Sphere(radius=marker_radius, center=robot_frame.tcp.point),
                color="yellow",
                name="animated_tcp_marker",
            )
            plotter.add_mesh(
                pv.Sphere(radius=marker_radius * 1.15, center=robot_frame.flange.point),
                color="white",
                name="animated_flange_marker",
            )
            add_frame_axes(robot_frame.tcp, "animated_tcp", ["red", "green", "blue"], 5)
            if self.show_flange_frames:
                add_frame_axes(robot_frame.flange, "animated_flange", ["magenta", "lime", "cyan"], 4)
            plotter.add_mesh(
                _line_polydata([np.array([robot_frame.flange.point, robot_frame.tcp.point])]),
                color="white",
                line_width=2,
                name="animated_tcp_flange_link",
            )
            plotter.add_text(
                f"Frame {index + 1}/{len(self.robot_frames)} | Speed {state['speed']:.1f}x",
                position="upper_left",
                font_size=10,
                color="white",
                name="animated_frame_text",
            )
            if tcp_data is not None:
                joint_text = ""
                if six_axis_pose is not None:
                    joint_angles_deg = np.degrees(six_axis_pose.joint_angles)
                    joint_text = (
                        "\n"
                        f"{format_joint_text(joint_angles_deg)}\n"
                        f"IK pos err: {six_axis_pose.position_error:.2f}\n"
                        f"IK rot err: {six_axis_pose.rotation_error_deg:.2f} deg"
                    )
                plotter.add_text(
                    (
                        "TCP vs WCS\n"
                        f"X: {tcp_data['X']:.3f}\n"
                        f"Y: {tcp_data['Y']:.3f}\n"
                        f"Z: {tcp_data['Z']:.3f}\n"
                        f"A: {tcp_data['A']:.2f} deg\n"
                        f"B: {tcp_data['B']:.2f} deg\n"
                        f"C: {tcp_data['C']:.2f} deg"
                        f"{joint_text}"
                    ),
                    position="lower_right",
                    font_size=10,
                    color="black",
                    name="animated_pose_text",
                )
            plotter.render()

        def show_interpolated_robot_pose(index: int, joint_angles: np.ndarray, substep: int, substeps: int) -> None:
            state["index"] = int(np.clip(index, 0, len(self.robot_frames) - 1))
            pose = pose_from_joint_angles(joint_angles)
            self.last_joint_angles = joint_angles.copy()
            state["jog_angles_deg"] = np.degrees(joint_angles).copy()
            state["wcs_jog_xyzabc"] = flange_xyzabc_from_transform(pose.transforms[-1])

            if not add_or_update_visual_robot_pose(pose):
                remove_animation_actors()
                add_six_axis_robot_pose(pose)
            else:
                clear_lightweight_pose_actors()
            plotter.add_text(
                (
                    f"Frame {state['index'] + 1}/{len(self.robot_frames)} | "
                    f"Smooth {substep}/{substeps} | Speed {state['speed']:.1f}x"
                ),
                position="upper_left",
                font_size=10,
                color="white",
                name="animated_frame_text",
            )

        def update_frame_cursor(value: float) -> None:
            if state["playing"]:
                return
            show_robot_frame(int(round(value)))

        def step_frame(delta: int) -> None:
            if state["playing"]:
                return
            next_index = int(np.clip(state["index"] + delta, 0, len(self.robot_frames) - 1))
            show_robot_frame(next_index)

        def previous_frame(_state: bool) -> None:
            step_frame(-1)

        def next_frame(_state: bool) -> None:
            step_frame(1)

        def build_playback_trajectory(start_index: int) -> List[Tuple[int, np.ndarray]]:
            home_angles = self.six_axis_robot.home_angles.copy()
            start_pose = machining_start_pose(home_angles)
            trajectory = [
                (-1, home_angles),
                (0, start_pose.joint_angles.copy()),
            ]
            previous_angles = start_pose.joint_angles.copy()
            path_start = max(0, start_index)
            for index in range(path_start, len(self.robot_frames), stride):
                if index == 0:
                    continue
                target_pose = self.six_axis_robot.solve_ik(
                    _frame_to_transform(self.robot_frames[index].flange),
                    previous_angles,
                )
                previous_angles = target_pose.joint_angles.copy()
                trajectory.append((index, previous_angles.copy()))

            final_index = len(self.robot_frames) - 1
            if trajectory and trajectory[-1][0] != final_index:
                target_pose = self.six_axis_robot.solve_ik(
                    _frame_to_transform(self.robot_frames[final_index].flange),
                    previous_angles,
                )
                trajectory.append((final_index, target_pose.joint_angles.copy()))

            return trajectory

        def expanded_playback_frames(trajectory: List[Tuple[int, np.ndarray]]) -> List[Tuple[int, np.ndarray, int, int]]:
            frames = []
            if not trajectory:
                return frames

            previous_index, previous_angles = trajectory[0]
            frames.append((previous_index, previous_angles.copy(), 0, interpolation_steps))
            for target_index, target_angles in trajectory[1:]:
                for substep in range(1, interpolation_steps + 1):
                    blend = substep / interpolation_steps
                    interpolated_angles = previous_angles + (target_angles - previous_angles) * blend
                    interpolated_index = int(round(previous_index + (target_index - previous_index) * blend))
                    frames.append((interpolated_index, interpolated_angles, substep, interpolation_steps))
                previous_index = target_index
                previous_angles = target_angles.copy()
            return frames

        def stop_playback(_state: bool = False) -> None:
            state["playing"] = False
            state["paused"] = False
            state["playback_generation"] += 1
            state["playback_cursor"] = 0
            state["playback_frames"] = []
            state["trajectory"] = None
            if plotter.actors.get("animated_frame_text"):
                plotter.remove_actor("animated_frame_text", render=False)
            plotter.add_text(
                "Playback stopped",
                position="upper_left",
                font_size=10,
                color="white",
                name="animated_frame_text",
            )
            plotter.render()

        def playback_timer_step(_timer_step: int, generation: int) -> None:
            if generation != state["playback_generation"]:
                return
            if not state["playing"] or state["paused"]:
                return

            frames = state["playback_frames"]
            cursor = state["playback_cursor"]
            if cursor >= len(frames):
                state["playing"] = False
                state["paused"] = False
                state["playback_cursor"] = 0
                return

            index, joint_angles, substep, substeps = frames[cursor]
            show_interpolated_robot_pose(index, joint_angles, substep, substeps)
            plotter.update()
            state["playback_cursor"] = cursor + 1

        def start_playback_timer(generation: int) -> None:
            remaining_frames = max(1, len(state["playback_frames"]) - state["playback_cursor"])
            timer_duration_ms = max(1, int(1000.0 / (target_fps * max(state["speed"], EPSILON))))
            plotter.add_timer_event(
                max_steps=max(remaining_frames * 20, 1000),
                duration=timer_duration_ms,
                callback=lambda timer_step: playback_timer_step(timer_step, generation),
            )

        def play_path(_state: bool) -> None:
            if state["playing"] and state["paused"]:
                state["paused"] = False
                state["playback_generation"] += 1
                start_playback_timer(state["playback_generation"])
                return

            if state["playing"]:
                state["paused"] = True
                state["playback_generation"] += 1
                if plotter.actors.get("animated_frame_text"):
                    plotter.remove_actor("animated_frame_text", render=False)
                plotter.add_text(
                    "Playback paused",
                    position="upper_left",
                    font_size=10,
                    color="white",
                    name="animated_frame_text",
                )
                plotter.render()
                return

            state["playing"] = True
            state["paused"] = False
            state["playback_generation"] += 1
            generation = state["playback_generation"]
            start_index = state["index"]
            plotter.add_text(
                "Preparing smooth robot path...",
                position="upper_left",
                font_size=10,
                color="white",
                name="animated_frame_text",
            )
            plotter.render()

            trajectory = build_playback_trajectory(start_index)
            state["trajectory"] = trajectory
            if not trajectory:
                state["playing"] = False
                return

            state["playback_frames"] = expanded_playback_frames(trajectory)
            state["playback_cursor"] = 0
            start_playback_timer(generation)

        def update_speed(value: float) -> None:
            state["speed"] = float(value)

        show_home_pose()
        plotter.reset_camera()
        if self.show_playback_buttons:
            add_momentary_button(
                lambda: play_path(False),
                position=(10, 10),
                size=30,
                color_on="lime",
                color_off="gray",
                background_color="black",
            )
            plotter.add_text(
                "Play/Pause",
                position=(48, 14),
                font_size=10,
                color="black",
                name="play_button_label",
            )
            add_momentary_button(
                lambda: stop_playback(False),
                position=(160, 10),
                size=30,
                color_on="tomato",
                color_off="gray",
                background_color="black",
            )
            plotter.add_text(
                "Stop",
                position=(198, 14),
                font_size=10,
                color="black",
                name="stop_button_label",
            )
            add_momentary_button(
                lambda: show_home_pose(False),
                position=(10, 54),
                size=30,
                color_on="deepskyblue",
                color_off="gray",
                background_color="black",
            )
            plotter.add_text(
                "Home",
                position=(48, 58),
                font_size=10,
                color="black",
                name="home_button_label",
            )
            add_momentary_button(
                lambda: show_machining_start_pose(False),
                position=(10, 98),
                size=30,
                color_on="orange",
                color_off="gray",
                background_color="black",
            )
            plotter.add_text(
                "Start",
                position=(48, 102),
                font_size=10,
                color="black",
                name="machining_start_button_label",
            )
            add_momentary_button(
                lambda: previous_frame(False),
                position=(700, 200),
                size=30,
                color_on="deepskyblue",
                color_off="gray",
                background_color="black",
            )
            plotter.add_text(
                "Back",
                position=(700, 250),
                font_size=10,
                color="black",
                name="previous_button_label",
            )
            add_momentary_button(
                lambda: next_frame(False),
                position=(850, 200),
                size=30,
                color_on="deepskyblue",
                color_off="gray",
                background_color="black",
            )
            plotter.add_text(
                "Forth",
                position=(850, 250),
                font_size=10,
                color="black",
                name="next_button_label",
            )
        plotter.add_slider_widget(
            update_speed,
            rng=(0.1, 5.0),
            value=state["speed"],
            title="Speed",
            pointa=(0.02, 0.10),
            pointb=(0.35, 0.10),
            style="modern",
        )
        plotter.add_slider_widget(
            update_frame_cursor,
            rng=(0, len(self.robot_frames) - 1),
            value=state["index"],
            title="Frame cursor",
            pointa=(0.42, 0.10),
            pointb=(0.95, 0.10),
            style="modern",
        )
        joint_slider_y = np.linspace(0.88, 0.52, joint_count)
        for joint_index, y_position in enumerate(joint_slider_y):
            lower_limit, upper_limit = self.six_axis_robot.joint_limits_deg[joint_index]
            plotter.add_slider_widget(
                lambda value, index=joint_index: update_joint_jog(index, value),
                rng=(lower_limit, upper_limit),
                value=state["jog_angles_deg"][joint_index],
                title=f"J{joint_index + 1}",
                pointa=(0.68, y_position),
                pointb=(0.98, y_position),
                style="modern",
            )
        sync_wcs_jog_from_current_robot()
        if self.show_playback_buttons:
            wcs_jog_controls = [
                ("X-", 0, -10.0, (10, 315), (48, 319)),
                ("X+", 0, 10.0, (92, 315), (130, 319)),
                ("Y-", 1, -10.0, (10, 355), (48, 359)),
                ("Y+", 1, 10.0, (92, 355), (130, 359)),
                ("Z-", 2, -10.0, (10, 395), (48, 399)),
                ("Z+", 2, 10.0, (92, 395), (130, 399)),
                ("A-", 3, -5.0, (10, 445), (48, 449)),
                ("A+", 3, 5.0, (92, 445), (130, 449)),
                ("B-", 4, -5.0, (10, 485), (48, 489)),
                ("B+", 4, 5.0, (92, 485), (130, 489)),
                ("C-", 5, -5.0, (10, 525), (48, 529)),
                ("C+", 5, 5.0, (92, 525), (130, 529)),
            ]
            plotter.add_text(
                "WCS jog",
                position=(10, 285),
                font_size=10,
                color="black",
                name="wcs_jog_label",
            )
            for label, axis_index, step, button_position, label_position in wcs_jog_controls:
                add_momentary_button(
                    lambda index=axis_index, delta=step: jog_wcs_axis(index, delta),
                    position=button_position,
                    size=28,
                    color_on="orange",
                    color_off="gray",
                    background_color="black",
                )
                plotter.add_text(
                    label,
                    position=label_position,
                    font_size=9,
                    color="black",
                    name=f"wcs_jog_{label.replace('+', 'plus').replace('-', 'minus')}_label",
                )
        plotter.add_key_event("Left", lambda: step_frame(-1))
        plotter.add_key_event("Right", lambda: step_frame(1))
        plotter.add_key_event("space", lambda: play_path(False))
        plotter.add_key_event("s", lambda: stop_playback(False))
        plotter.add_key_event("h", lambda: show_home_pose(False))
        plotter.add_key_event("m", lambda: show_machining_start_pose(False))
        if self.auto_start_playback:
            plotter.add_timer_event(
                max_steps=1,
                duration=700,
                callback=lambda _timer_step: play_path(False),
            )

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
                position="lower_left",
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


SCRIPT_DIR = Path(__file__).resolve().parent
KR10_R1100_URDF_PATH = SCRIPT_DIR / "robot_models" / "kuka_experimental" / "kuka_kr10_support" / "urdf" / "kr10r1100sixx_pybullet.urdf"

# Edit these six values to define the KUKA KR10 R1100 sixx arm home. The 7th axis is the rotary table below.
KR10_R1100_HOME_JOINT_ANGLES_DEG = np.array([0.0, -90.0, 90.0, 0.0, 0.0, 0.0], dtype=float)

# Edit this to rotate the workobject/table around its local Z axis before slicing.
ROTARY_TABLE_CENTER_XYZ = np.array([300.0, 200.0, 200.0], dtype=float)
ROTARY_TABLE_ANGLE_DEG = 0.0

# Use "top" for only the top section, or "layers" for the old full-height slicing.
TOOLPATH_MODE = "top"
TOP_SECTION_OFFSET = 0.5

# Keep this button-free if PyVista widget clicks behave badly on your machine.
AUTO_START_SIMULATION = True
SHOW_PLAYBACK_BUTTONS = False


def main() -> None:
    visualizer = MeshVisualizer(
        mesh_path=r"C:\Users\shish\C-\Slicing\Ramp.stl",
        wcs_origin=np.array([0, 0, 0]),
        stl_target_position=np.array([300, 200, 200]),
        stl_target_abc=np.array([0.0, 0.0, 0]),
        num_layers=30,
        axis_length=10,
        toolpath_clearance=50.0,
        toolpath_mode=TOOLPATH_MODE,
        top_section_offset=TOP_SECTION_OFFSET,
        tcp_offset_xyzabc=np.array([0.0, 0.0, 500, 0, 0.0, 0.0]),
        robot_kinematics_config=SixAxisRobotConfig(
            urdf_path=str(KR10_R1100_URDF_PATH),
            end_effector_link_name="link_6",
            active_joint_count=6,
            base_xyzabc=np.array([0, 0, 0, 0.0, 0.0, 0]),
            home_joint_angles_deg=KR10_R1100_HOME_JOINT_ANGLES_DEG,
        ),
        rotary_table_config=RotaryTableConfig(
            enabled=True,
            center_xyz=ROTARY_TABLE_CENTER_XYZ,
            axis_xyz=np.array([0.0, 0.0, 1.0], dtype=float),
            angle_deg=ROTARY_TABLE_ANGLE_DEG,
            radius=160.0,
            height=35.0,
        ),
        show_playback_buttons=SHOW_PLAYBACK_BUTTONS,
        auto_start_playback=AUTO_START_SIMULATION,
        show_flange_frames=False #flange frame visibility
    )

    visualizer.generate_path_data()
    visualizer.generate_frames()
    visualizer.generate_6dof_data()

    if visualizer.toolpath_data:
        visualizer.print_6dof_data_for_index(len(visualizer.toolpath_data) // 2)

    print("Simulation controls: Space=play/pause, S=stop, Left/Right=step, H=home, M=machining start.")
    visualizer.get_scene().show()


if __name__ == "__main__":
    main()
