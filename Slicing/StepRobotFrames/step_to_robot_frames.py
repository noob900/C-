#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


try:
    from OCP.BRep import BRep_Tool
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.BRepTools import BRepTools
    from OCP.GeomLProp import GeomLProp_SLProps
    from OCP.gp import gp_Pnt2d
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_OUT, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
except ImportError as import_error:
    BRep_Tool = None
    BRepClass_FaceClassifier = None
    BRepTools = None
    GeomLProp_SLProps = None
    gp_Pnt2d = None
    IFSelect_RetDone = None
    STEPControl_Reader = None
    TopAbs_FACE = None
    TopAbs_OUT = None
    TopAbs_REVERSED = None
    TopExp_Explorer = None
    OCP_IMPORT_ERROR = import_error
else:
    OCP_IMPORT_ERROR = None


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, value: float) -> "Vec3":
        return Vec3(self.x * value, self.y * value, self.z * value)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def length_squared(self) -> float:
        return self.dot(self)

    def normalized(self) -> "Vec3":
        length = self.length()
        if length <= 1e-12:
            raise ValueError("Cannot normalize a zero-length vector.")
        return self * (1.0 / length)


@dataclass(frozen=True)
class Frame:
    index: int
    face_index: int
    surface_point: Vec3
    normal: Vec3
    origin: Vec3
    x_axis: Vec3
    y_axis: Vec3
    z_axis: Vec3
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    rx_rad: float
    ry_rad: float
    rz_rad: float


def require_ocp() -> None:
    if OCP_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing OpenCascade Python bindings.\n"
            "Install dependencies with:\n"
            "  python -m pip install -r requirements.txt\n\n"
            f"Original import error: {OCP_IMPORT_ERROR}"
        )


def load_step_shape(step_path: Path):
    require_ocp()

    if not step_path.exists():
        raise RuntimeError(f"STEP file not found: {step_path}")

    # Check for valid STEP header to catch non-STEP files or OneDrive placeholders
    try:
        with open(step_path, 'rb') as f:
            content_start = f.read(1024)
            if b"QUID" in content_start:
                raise RuntimeError(f"File '{step_path.name}' is a OneDrive placeholder. Right-click it and select 'Always keep on this device'.")
            if b"ISO-10303-21" not in content_start and b"HEADER" not in content_start:
                raise RuntimeError(f"File '{step_path.name}' is not a valid STEP file (missing ISO-10303-21 header).")
    except (IOError, OSError) as e:
        print(f"Warning: Pre-check of file header failed: {e}")

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"OpenCASCADE failed to parse STEP file: {step_path}. Ensure it is a valid AP203/214/242 file.")

    transferred = reader.TransferRoots()
    if transferred == 0:
        raise RuntimeError(f"No transferable STEP roots found in: {step_path}")

    return reader.OneShape()


def iter_faces(shape) -> Iterable:
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        yield as_face(explorer.Current())
        explorer.Next()


def as_face(shape):
    try:
        from OCP.TopoDS import topods

        return topods.Face(shape)
    except Exception:
        try:
            from OCP.TopoDS import TopoDS

            return TopoDS.Face_s(shape)
        except Exception:
            return shape


def uv_bounds(face) -> Tuple[float, float, float, float]:
    try:
        return tuple(BRepTools.UVBounds_s(face))
    except Exception:
        try:
            return tuple(BRepTools.UVBounds(face))
        except Exception as exc:
            raise RuntimeError("Could not read face UV bounds.") from exc


def face_surface(face):
    try:
        return BRep_Tool.Surface_s(face)
    except Exception:
        return BRep_Tool.Surface(face)


def linspace(start: float, end: float, count: int) -> List[float]:
    if count <= 1:
        return [(start + end) * 0.5]
    return [start + (end - start) * (i / float(count - 1)) for i in range(count)]


def valid_finite_bounds(bounds: Sequence[float]) -> bool:
    return all(math.isfinite(value) and abs(value) < 1e100 for value in bounds)


def uv_inside_face(face, u: float, v: float, tolerance: float) -> bool:
    try:
        classifier = BRepClass_FaceClassifier(face, gp_Pnt2d(u, v), tolerance)
        return classifier.State() != TopAbs_OUT
    except Exception:
        return True


def point_and_normal(face, surface, u: float, v: float, tolerance: float) -> Optional[Tuple[Vec3, Vec3]]:
    if not uv_inside_face(face, u, v, tolerance):
        return None

    try:
        props = GeomLProp_SLProps(surface, u, v, 1, tolerance)
        if not props.IsNormalDefined():
            return None

        point = props.Value()
        normal = props.Normal()
        normal_vec = Vec3(normal.X(), normal.Y(), normal.Z()).normalized()

        try:
            if face.Orientation() == TopAbs_REVERSED:
                normal_vec = normal_vec * -1.0
        except Exception:
            pass

        return Vec3(point.X(), point.Y(), point.Z()), normal_vec
    except Exception:
        return None


def matrix_from_axes(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> List[List[float]]:
    return [
        [x_axis.x, y_axis.x, z_axis.x],
        [x_axis.y, y_axis.y, z_axis.y],
        [x_axis.z, y_axis.z, z_axis.z],
    ]


def rpy_zyx_degrees(rotation: List[List[float]]) -> Tuple[float, float, float]:
    r00, _, _ = rotation[0]
    r10, _, _ = rotation[1]
    r20, r21, r22 = rotation[2]

    if abs(r20) < 1.0 - 1e-9:
        pitch = math.asin(-r20)
        roll = math.atan2(r21, r22)
        yaw = math.atan2(r10, r00)
    else:
        pitch = math.pi / 2.0 if r20 <= -1.0 else -math.pi / 2.0
        roll = 0.0
        yaw = math.atan2(-rotation[0][1], rotation[1][1])

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def axis_angle(rotation: List[List[float]]) -> Tuple[float, float, float]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    angle = math.acos(cos_angle)

    if abs(angle) < 1e-12:
        return 0.0, 0.0, 0.0

    sin_angle = math.sin(angle)
    if abs(sin_angle) < 1e-12:
        return 0.0, 0.0, angle

    axis = Vec3(
        rotation[2][1] - rotation[1][2],
        rotation[0][2] - rotation[2][0],
        rotation[1][0] - rotation[0][1],
    ) * (1.0 / (2.0 * sin_angle))

    return axis.x * angle, axis.y * angle, axis.z * angle


def reference_axis(name: str) -> Vec3:
    if name == "world_y":
        return Vec3(0.0, 1.0, 0.0)
    if name == "world_z":
        return Vec3(0.0, 0.0, 1.0)
    return Vec3(1.0, 0.0, 0.0)


def build_frame(
    index: int,
    face_index: int,
    surface_point: Vec3,
    normal: Vec3,
    tool_axis: str,
    ref_axis_name: str,
    stand_off: float,
) -> Frame:
    normal = normal.normalized()
    z_axis = (normal * -1.0 if tool_axis == "toward_surface" else normal).normalized()

    ref = reference_axis(ref_axis_name)
    if abs(ref.dot(z_axis)) > 0.95:
        ref = Vec3(0.0, 1.0, 0.0)
    if abs(ref.dot(z_axis)) > 0.95:
        ref = Vec3(0.0, 0.0, 1.0)

    x_axis = (ref - z_axis * ref.dot(z_axis)).normalized()
    y_axis = z_axis.cross(x_axis).normalized()
    x_axis = y_axis.cross(z_axis).normalized()

    origin = surface_point + normal * stand_off
    rotation = matrix_from_axes(x_axis, y_axis, z_axis)
    roll_deg, pitch_deg, yaw_deg = rpy_zyx_degrees(rotation)
    rx_rad, ry_rad, rz_rad = axis_angle(rotation)

    return Frame(
        index=index,
        face_index=face_index,
        surface_point=surface_point,
        normal=normal,
        origin=origin,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        rx_rad=rx_rad,
        ry_rad=ry_rad,
        rz_rad=rz_rad,
    )


def sort_frames_nearest_neighbor(frames: List[Frame]) -> List[Frame]:
    """Sorts frames to minimize the travel distance between consecutive points."""
    if not frames:
        return []
    
    unvisited = list(frames)
    sorted_path = [unvisited.pop(0)]
    
    while unvisited:
        last_point = sorted_path[-1].origin
        nearest_idx = min(range(len(unvisited)), key=lambda i: (unvisited[i].origin - last_point).length_squared())
        sorted_path.append(unvisited.pop(nearest_idx))
        
    return sorted_path


def generate_frames(
    step_path: Path,
    samples_u: int,
    samples_v: int,
    trim_margin: float,
    tolerance: float,
    tool_axis: str,
    ref_axis_name: str,
    stand_off: float,
    max_points: Optional[int],
    optimize_path: bool = True,
) -> List[Frame]:
    shape = load_step_shape(step_path)
    frames: List[Frame] = []

    for face_index, face in enumerate(iter_faces(shape), start=1):
        bounds = uv_bounds(face)
        if not valid_finite_bounds(bounds):
            continue

        u_min, u_max, v_min, v_max = bounds
        if u_max <= u_min or v_max <= v_min:
            continue

        u_margin = (u_max - u_min) * trim_margin
        v_margin = (v_max - v_min) * trim_margin
        u_values = linspace(u_min + u_margin, u_max - u_margin, samples_u)
        v_values = linspace(v_min + v_margin, v_max - v_margin, samples_v)
        surface = face_surface(face)

        for u in u_values:
            for v in v_values:
                point_normal = point_and_normal(face, surface, u, v, tolerance)
                if point_normal is None:
                    continue

                point, normal = point_normal
                frames.append(
                    build_frame(
                        index=len(frames) + 1,
                        face_index=face_index,
                        surface_point=point,
                        normal=normal,
                        tool_axis=tool_axis,
                        ref_axis_name=ref_axis_name,
                        stand_off=stand_off,
                    )
                )

                if max_points is not None and len(frames) >= max_points:
                    break
        if max_points is not None and len(frames) >= max_points:
            break

    return sort_frames_nearest_neighbor(frames) if optimize_path else frames


def frame_row(frame: Frame) -> dict:
    return {
        "index": frame.index,
        "face_index": frame.face_index,
        "surface_x": frame.surface_point.x,
        "surface_y": frame.surface_point.y,
        "surface_z": frame.surface_point.z,
        "normal_x": frame.normal.x,
        "normal_y": frame.normal.y,
        "normal_z": frame.normal.z,
        "origin_x": frame.origin.x,
        "origin_y": frame.origin.y,
        "origin_z": frame.origin.z,
        "x_axis_x": frame.x_axis.x,
        "x_axis_y": frame.x_axis.y,
        "x_axis_z": frame.x_axis.z,
        "y_axis_x": frame.y_axis.x,
        "y_axis_y": frame.y_axis.y,
        "y_axis_z": frame.y_axis.z,
        "z_axis_x": frame.z_axis.x,
        "z_axis_y": frame.z_axis.y,
        "z_axis_z": frame.z_axis.z,
        "roll_deg": frame.roll_deg,
        "pitch_deg": frame.pitch_deg,
        "yaw_deg": frame.yaw_deg,
        "rx_rad": frame.rx_rad,
        "ry_rad": frame.ry_rad,
        "rz_rad": frame.rz_rad,
    }


def write_csv(frames: Sequence[Frame], out_path: Path) -> None:
    rows = [frame_row(frame) for frame in frames]
    if not rows:
        raise RuntimeError("No frames were generated.")

    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_urscript(frames: Sequence[Frame], out_path: Path, unit_scale: float, speed: float, accel: float) -> None:
    with out_path.open("w", encoding="utf-8") as file:
        file.write("def machining_path():\n")
        for frame in frames:
            x = frame.origin.x * unit_scale
            y = frame.origin.y * unit_scale
            z = frame.origin.z * unit_scale
            file.write(
                "  movel(p[{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}], a={:.6f}, v={:.6f})\n".format(
                    x,
                    y,
                    z,
                    frame.rx_rad,
                    frame.ry_rad,
                    frame.rz_rad,
                    accel,
                    speed,
                )
            )
        file.write("end\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    # Hardcoded path inside the code
    default_step_path = Path(r"C:\Users\shish\OneDrive\Desktop\CAD\Cube.step")

    parser = argparse.ArgumentParser(
        description="Read a STEP file and generate WCS machining frames for a 6-axis robot."
    )
    parser.add_argument(
        "step_file", 
        type=Path, 
        nargs="?", 
        default=default_step_path, 
        help=f"Input STEP file path. Defaults to: {default_step_path}")
    parser.add_argument("--out", type=Path, default=Path("robot_frames.csv"), help="Output CSV path.")
    parser.add_argument("--samples-u", type=int, default=8, help="Sample count in each face U direction.")
    parser.add_argument("--samples-v", type=int, default=8, help="Sample count in each face V direction.")
    parser.add_argument("--max-points", type=int, default=None, help="Optional maximum number of frames to export.")
    parser.add_argument("--stand-off", type=float, default=0.0, help="Offset origin along outward surface normal.")
    parser.add_argument("--trim-margin", type=float, default=0.02, help="Fractional UV margin to avoid face edges.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Geometry tolerance.")
    parser.add_argument(
        "--tool-axis",
        choices=("toward_surface", "away_from_surface"),
        default="toward_surface",
        help="Direction of frame Z axis relative to the sampled surface normal.",
    )
    parser.add_argument(
        "--reference-axis",
        choices=("world_x", "world_y", "world_z"),
        default="world_x",
        help="Preferred WCS axis used to stabilize frame X orientation.",
    )
    parser.add_argument("--urscript", type=Path, default=None, help="Optional Universal Robots script output path.")
    parser.add_argument("--unit-scale", type=float, default=1.0, help="Scale positions for robot export.")
    parser.add_argument("--ur-speed", type=float, default=0.05, help="URScript movel speed.")
    parser.add_argument("--ur-accel", type=float, default=0.2, help="URScript movel acceleration.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.step_file.exists():
        print(f"STEP file not found: {args.step_file}", file=sys.stderr)
        return 2

    frames = generate_frames(
        step_path=args.step_file,
        samples_u=max(1, args.samples_u),
        samples_v=max(1, args.samples_v),
        trim_margin=max(0.0, min(0.45, args.trim_margin)),
        tolerance=args.tolerance,
        tool_axis=args.tool_axis,
        ref_axis_name=args.reference_axis,
        stand_off=args.stand_off,
        max_points=args.max_points,
    )

    write_csv(frames, args.out)
    if args.urscript is not None:
        write_urscript(frames, args.urscript, args.unit_scale, args.ur_speed, args.ur_accel)

    print(f"Generated {len(frames)} robot frames.")
    print(f"CSV: {args.out.resolve()}")
    if args.urscript is not None:
        print(f"URScript: {args.urscript.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
