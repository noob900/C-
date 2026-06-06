import numpy as np
import math
from pathlib import Path
from typing import List, Union, Optional
import step_to_robot_frames as srf

class CADProcessor:
    """
    A class to load CAD models (STEP/STL), process them into robotic frames,
    and perform spatial transformations for path planning.
    """
    def __init__(self):
        self.frames: List[srf.Frame] = []

    def load_and_process(self, file_path: Union[str, Path], **kwargs) -> List[srf.Frame]:
        """
        Loads a CAD file and generates toolpath frames.
        Supported extensions: .step, .stp, .stl
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CAD file not found: {path}")
            
        # Heuristic: STEP files under 1KB are likely OneDrive placeholders or empty
        if path.stat().st_size < 1024:
            print(f"Warning: File {path.name} is very small ({path.stat().st_size} bytes). "
                  "It might be a OneDrive placeholder.")

        ext = path.suffix.lower()
        if ext in [".step", ".stp"]:
            # Leverage existing STEP processing
            self.frames = srf.generate_frames(
                step_path=path,
                samples_u=kwargs.get("samples_u", 8),
                samples_v=kwargs.get("samples_v", 8),
                trim_margin=kwargs.get("trim_margin", 0.02),
                tolerance=kwargs.get("tolerance", 1e-6),
                tool_axis=kwargs.get("tool_axis", "toward_surface"),
                ref_axis_name=kwargs.get("ref_axis_name", "world_x"),
                stand_off=kwargs.get("stand_off", 0.0),
                max_points=kwargs.get("max_points", None),
                optimize_path=kwargs.get("optimize_path", True)
            )
        elif ext == ".stl":
            # Placeholder for STL sampling logic (Mesh processing)
            print("STL format detected. Mesh sampling implementation would go here.")
            self.frames = []
        else:
            raise ValueError(f"Unsupported extension: {ext}")

        return self.frames

    def apply_global_transform(self, rotation_deg: List[float], translation: List[float]):
        """
        Transforms all frames by a global rotation (Euler XYZ) and translation.
        This aligns the CAD coordinates with the Robot's base or User Frame.
        """
        if not self.frames:
            return

        # Convert degrees to radians
        rad = [math.radians(d) for d in rotation_deg]
        
        # Component rotation matrices
        cx, sx = math.cos(rad[0]), math.sin(rad[0])
        cy, sy = math.cos(rad[1]), math.sin(rad[1])
        cz, sz = math.cos(rad[2]), math.sin(rad[2])

        # Combined Rotation matrix R = Rz * Ry * Rx
        R = np.array([
            [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz],
            [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
            [-sy,   sx*cy,           cx*cy]
        ])
        T = np.array(translation)

        transformed_frames = []
        for f in self.frames:
            # 1. Transform points (Origin and Surface Point)
            new_orig_arr = R @ np.array([f.origin.x, f.origin.y, f.origin.z]) + T
            new_surf_arr = R @ np.array([f.surface_point.x, f.surface_point.y, f.surface_point.z]) + T
            
            # 2. Transform orientation vectors (Rotation only)
            new_x_vec = R @ np.array([f.x_axis.x, f.x_axis.y, f.x_axis.z])
            new_y_vec = R @ np.array([f.y_axis.x, f.y_axis.y, f.y_axis.z])
            new_z_vec = R @ np.array([f.z_axis.x, f.z_axis.y, f.z_axis.z])
            new_norm_vec = R @ np.array([f.normal.x, f.normal.y, f.normal.z])

            # 3. Convert back to Vec3 and recalculate orientation angles
            vx, vy, vz = srf.Vec3(*new_x_vec), srf.Vec3(*new_y_vec), srf.Vec3(*new_z_vec)
            rot_matrix = srf.matrix_from_axes(vx, vy, vz)
            roll, pitch, yaw = srf.rpy_zyx_degrees(rot_matrix)
            rx, ry, rz = srf.axis_angle(rot_matrix)

            transformed_frames.append(srf.Frame(
                index=f.index,
                face_index=f.face_index,
                surface_point=srf.Vec3(*new_surf_arr),
                normal=srf.Vec3(*new_norm_vec),
                origin=srf.Vec3(*new_orig_arr),
                x_axis=vx, y_axis=vy, z_axis=vz,
                roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw,
                rx_rad=rx, ry_rad=ry, rz_rad=rz
            ))
        
        self.frames = transformed_frames

    def get_as_array(self) -> np.ndarray:
        """
        Returns a Nx6 numpy array: [x, y, z, rx_rad, ry_rad, rz_rad]
        Optimized for numerical analysis and inverse kinematics.
        """
        return np.array([
            [f.origin.x, f.origin.y, f.origin.z, f.rx_rad, f.ry_rad, f.rz_rad]
            for f in self.frames
        ])

if __name__ == "__main__":
    # Initialization
    processor = CADProcessor()
    # Update this path to your local CAD model
    step_path = Path(r"C:\Users\shish\OneDrive\Desktop\CAD\LeadScrew Nut 8mm x 2mmPitch.STEP")
    
    if step_path.exists():
        print(f"--- Step 1: Loading and Processing CAD ---")
        processor.load_and_process(step_path, samples_u=5, samples_v=5)
        print(f"Frames generated: {len(processor.frames)}")
        
        print(f"\n--- Step 2: Applying Spatial Orientation ---")
        # Example: Rotate 90 deg around Z and shift 100mm on X-axis
        processor.apply_global_transform(rotation_deg=[0, 0, 90], translation=[100, 0, 0])
        
        print(f"\n--- Step 3: Storing in NumPy Array ---")
        path_array = processor.get_as_array()
        print(f"Array Shape: {path_array.shape}")
        print("Sample Frame Data (First Point):\n", path_array[0])
    else:
        print(f"Please ensure {step_path} exists to run the example.")