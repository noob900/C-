import numpy as np
from scipy.interpolate import splprep, splev
import trimesh
from typing import Tuple, List, Dict
import math

import os # Added for creating output directory

def _create_wcs_origin(origin_point=np.array([0, 0, 0]), axis_length=50.0):
    """
    Create a World Coordinate System (WCS) origin visualization.
    
    Args:
        origin_point: 3D position of the WCS origin [x, y, z]
        axis_length: Length of the axis lines
    
    Returns:
        tuple: (x_axis, y_axis, z_axis, origin_sphere) - geometry objects for visualization
        - Red line for X axis
        - Green line for Y axis
        - Blue line for Z axis
        - Gray sphere at origin
    """
    origin_point = np.array(origin_point, dtype=float)
    
    # Create axis endpoints
    x_end = origin_point + np.array([axis_length, 0, 0], dtype=float)
    y_end = origin_point + np.array([0, axis_length, 0], dtype=float)
    z_end = origin_point + np.array([0, 0, axis_length], dtype=float)
    
    # Create line geometries for each axis
    x_axis_points = np.array([origin_point, x_end], dtype=float)
    y_axis_points = np.array([origin_point, y_end], dtype=float)
    z_axis_points = np.array([origin_point, z_end], dtype=float)
    
    # Create Path3D objects for each axis
    x_axis = trimesh.load_path(x_axis_points)
    y_axis = trimesh.load_path(y_axis_points)
    z_axis = trimesh.load_path(z_axis_points)
    
    # Set colors per-entity: Red for X, Green for Y, Blue for Z
    x_axis.colors = np.array([[255, 0, 0, 255]], dtype=np.uint8)  # Red
    y_axis.colors = np.array([[0, 255, 0, 255]], dtype=np.uint8)  # Green
    z_axis.colors = np.array([[0, 0, 255, 255]], dtype=np.uint8)  # Blue
    
    # Create a small sphere at the origin to mark the WCS point
    origin_sphere = trimesh.creation.icosphere(subdivisions=2, radius=axis_length * 0.1)
    origin_sphere.apply_translation(origin_point)
    origin_sphere.visual.vertex_colors = [128, 128, 128, 255]  # Gray
    
    return x_axis, y_axis, z_axis, origin_sphere

def _normal_to_rpy(normal_vec: np.ndarray, ref_axis_name: str = "world_x") -> Tuple[float, float, float]:
    """
    Converts a normal vector (which becomes the tool's Z-axis) into Roll, Pitch, Yaw Euler angles (degrees).
    A reference axis is used to stabilize the tool's X-axis.
    The convention for Euler angles is ZYX (Yaw, Pitch, Roll).
    
    Args:
        normal_vec: A 3D numpy array representing the normal vector.
        ref_axis_name: String indicating the preferred world axis for stabilizing the tool's X-axis.
                       Choices: "world_x", "world_y", "world_z".
                       
    Returns:
        Tuple[float, float, float]: Roll, Pitch, Yaw angles in degrees.
    """
    normal_vec = normal_vec / np.linalg.norm(normal_vec) # Ensure normalized
    z_axis = normal_vec

    # Define a reference vector for stabilizing the X-axis
    if ref_axis_name == "world_y":
        ref = np.array([0.0, 1.0, 0.0])
    elif ref_axis_name == "world_z":
        ref = np.array([0.0, 0.0, 1.0])
    else: # default to world_x
        ref = np.array([1.0, 0.0, 0.0])

    # If the normal is too close to the reference, pick another reference
    # This avoids issues where cross product would be near zero
    if np.abs(np.dot(ref, z_axis)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    if np.abs(np.dot(ref, z_axis)) > 0.95:
        ref = np.array([0.0, 0.0, 1.0])

    # Calculate X-axis: orthogonal to Z-axis and as close to ref as possible
    x_axis = ref - z_axis * np.dot(ref, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Calculate Y-axis: orthogonal to both X and Z
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Re-orthogonalize X-axis to ensure perfect orthogonality
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Construct rotation matrix from axes (columns are X, Y, Z axes of the new frame)
    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T

    # Convert rotation matrix to ZYX Euler angles (Yaw, Pitch, Roll)
    r20 = rotation_matrix[2,0]
    if abs(r20) < 1.0 - 1e-9: # Not at gimbal lock
        pitch = math.asin(-r20)
        roll = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
        yaw = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
    else: # Gimbal lock
        pitch = math.pi / 2.0 if r20 <= -1.0 else -math.pi / 2.0
        roll = 0.0 # Arbitrarily set roll to 0
        yaw = math.atan2(-rotation_matrix[0,1], rotation_matrix[1,1]) # Yaw + roll is determined, but not individually

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

def _create_batch_point_frames(points, axis_length=2.0):
    """
    Creates small RGB coordinate frames at each provided point efficiently.
    """
    if len(points) == 0:
        return []
        
    # Create line segments for each axis
    x_segs = []
    y_segs = []
    z_segs = []
    
    for p in points:
        x_segs.append([p, p + np.array([axis_length, 0, 0])])
        y_segs.append([p, p + np.array([0, axis_length, 0])])
        z_segs.append([p, p + np.array([0, 0, axis_length])])
        
    # Load as Path3D objects
    x_path = trimesh.load_path(x_segs)
    x_path.colors = np.full((len(x_segs), 4), [255, 0, 0, 180], dtype=np.uint8)
    
    y_path = trimesh.load_path(y_segs)
    y_path.colors = np.full((len(y_segs), 4), [0, 255, 0, 180], dtype=np.uint8)
    
    z_path = trimesh.load_path(z_segs)
    z_path.colors = np.full((len(z_segs), 4), [0, 0, 255, 180], dtype=np.uint8)
    
    return [x_path, y_path, z_path]

class MeshVisualizer:
    """
    A class to load a mesh, generate toolpath points and paths,
    and create a 3D visualization of the process.
    """
    def __init__(self, mesh_path: str, wcs_origin: np.ndarray, stl_target_position: np.ndarray,
                 num_layers: int, sphere_radius: float, axis_length: float, path_thickness: float):
        # Store configuration
        self.mesh_path = mesh_path
        self.wcs_origin = np.array(wcs_origin, dtype=float)
        self.stl_target_position = np.array(stl_target_position, dtype=float)
        self.num_layers = num_layers
        self.sphere_radius = sphere_radius
        self.path_thickness = path_thickness
        self.axis_length = axis_length

        # Initialize state
        self.mesh: trimesh.Trimesh = None
        self.scene_geometries: List[trimesh.Trimesh] = []
        
        # Data generated by processing
        self.points: np.ndarray = np.array([])
        self.normals: np.ndarray = np.array([])
        self.green_paths: List[np.ndarray] = []
        self.red_paths: List[np.ndarray] = []
        self.toolpath_data: List[Dict] = []

        # Load and position the mesh upon initialization
        self._load_and_position_mesh()

    def _load_and_position_mesh(self):
        """Loads the mesh from file and translates it to the target position."""
        self.mesh = trimesh.load(self.mesh_path, force='mesh')
        mesh_centroid = self.mesh.centroid.copy()
        translation_vector = self.stl_target_position - mesh_centroid
        self.mesh.apply_translation(translation_vector)
        print(f"Loaded and positioned mesh. Centroid at: {self.mesh.centroid}")

    def generate_path_data(self, smoothing=0.1):
        """Generates points, normals, and paths from the mesh slices."""
        if self.mesh is None:
            raise ValueError("Mesh has not been loaded. Cannot generate paths.")

        path_geometries, self.green_paths, self.red_paths, green_points_data, red_points_data, green_normals_data, red_normals_data = generate_smooth_layered_path_from_mesh(
            self.mesh, num_layers=self.num_layers, smoothing=smoothing, line_thickness=self.path_thickness
        )
        self.scene_geometries.extend(path_geometries)

        # Consolidate all generated points and normals
        all_points = []
        if green_points_data:
            all_points.extend(np.vstack(green_points_data))
        if red_points_data:
            all_points.extend(np.vstack(red_points_data))
        
        all_normals = []
        if green_normals_data:
            all_normals.extend(np.vstack(green_normals_data))
        if red_normals_data:
            all_normals.extend(np.vstack(red_normals_data))

        self.points = np.array(all_points)
        self.normals = np.array(all_normals)
        print(f"Generated {len(self.points)} points and {len(self.normals)} normals.")

    def visualize_points_as_spheres(self):
        """Creates a single mesh of spheres at each generated point."""
        if len(self.points) == 0:
            print("No points to visualize.")
            return

        proto = trimesh.creation.icosphere(subdivisions=1, radius=self.sphere_radius)
        v_proto, f_proto = proto.vertices, proto.faces
        n_v, n_f, n_p = len(v_proto), len(f_proto), len(self.points)

        all_v = np.tile(v_proto, (n_p, 1)) + np.repeat(self.points, n_v, axis=0)
        offsets = np.arange(n_p) * n_v
        all_f = np.tile(f_proto, (n_p, 1)) + np.repeat(offsets, n_f)[:, None]

        combined_spheres = trimesh.Trimesh(vertices=all_v, faces=all_f, process=False)
        combined_spheres.visual.face_colors = [0, 0, 0, 255]  # Black spheres
        self.scene_geometries.append(combined_spheres)
        print("Generated sphere visualization for points.")

    def visualize_coordinate_frames(self):
        """Visualizes WCS, STL origin, and a sample of point frames."""
        # WCS visualization
        x_axis, y_axis, z_axis, origin_sphere = _create_wcs_origin(self.wcs_origin, self.axis_length)
        self.scene_geometries.extend([x_axis, y_axis, z_axis, origin_sphere])

        # STL reference point visualization (magenta axes)
        stl_x, stl_y, stl_z, stl_sphere = _create_wcs_origin(self.stl_target_position, self.axis_length * 0.5)
        stl_x.colors = np.array([[255, 0, 255, 255]], dtype=np.uint8)
        stl_y.colors = np.array([[255, 0, 255, 255]], dtype=np.uint8)
        stl_z.colors = np.array([[255, 0, 255, 255]], dtype=np.uint8)
        stl_sphere.visual.vertex_colors = [255, 0, 255, 255]
        self.scene_geometries.extend([stl_x, stl_y, stl_z, stl_sphere])

        # Sampled point frames
        if len(self.points) > 0:
            num_frames = min(100, len(self.points))
            sample_indices = np.linspace(0, len(self.points) - 1, num_frames, dtype=int)
            sampled_points = self.points[sample_indices]
            point_frames = _create_batch_point_frames(sampled_points, axis_length=self.axis_length * 0.2)
            self.scene_geometries.extend(point_frames)
        
        print("Generated coordinate frame visualizations.")

    def generate_6dof_data(self):
        """Generates a list of 6-DOF toolpath data dictionaries."""
        if len(self.points) != len(self.normals) or len(self.points) == 0:
            print("Warning: Mismatch between points and normals, or no data. 6-DOF data not generated.")
            return

        for i in range(len(self.points)):
            point = self.points[i]
            normal = self.normals[i]

            relative_xyz = point - self.wcs_origin
            roll_deg, pitch_deg, yaw_deg = _normal_to_rpy(normal)

            self.toolpath_data.append({
                'X': relative_xyz[0], 'Y': relative_xyz[1], 'Z': relative_xyz[2],
                'A': roll_deg, 'B': pitch_deg, 'C': yaw_deg
            })
        print(f"Generated {len(self.toolpath_data)} 6-DOF toolpath entries.")

    def get_scene(self) -> trimesh.Scene:
        """Returns the final scene object with all generated geometries."""
        return trimesh.Scene(self.scene_geometries)

    def export_visualization(self, output_path="outputs/visualization.html"):
        """Exports the current scene to an HTML file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            scene = self.get_scene()
            with open(output_path, 'w') as f:
                f.write(scene.export(file_type='gltf'))
            print(f"Visualization exported to {output_path}")
        except Exception as e:
            print(f"Warning: Could not export visualization to HTML: {e}")


def generate_smooth_layered_path_from_mesh(mesh, num_layers=30, smoothing=0.1, line_thickness=0.05):
    """ 
    Generates a continuous layered toolpath from an already-loaded mesh.
    (Similar to generate_smooth_layered_path but takes a mesh object instead of path)
    """
    # Use solid gray. Transparency (alpha < 255) significantly slows down the 3D renderer.
    mesh.visual.face_colors = np.full((len(mesh.faces), 4), [180, 180, 180, 255], dtype=np.uint8)

    bounds = mesh.bounds
    extents = mesh.extents
    model_scale = np.max(extents)
    # Generate continuous Z increments from bottom to top
    z_levels = np.linspace(bounds[0][2], bounds[1][2], num_layers)
    
    geometries = [] # Mesh is not added here to keep the view clean for points/paths
    last_layer_end_point = None
    green_paths = []
    red_paths = []
    green_points_data = []
    red_points_data = []
    green_normals_data = []  # New: to store normals for successful paths
    red_normals_data = []    # New: to store normals for fallback paths
    forward_direction = True

    triangles = mesh.triangles
    face_normals = mesh.face_normals

    print(f"Slicing model boundaries from Min Z={bounds[0][2]:.2f} to Max Z={bounds[1][2]:.2f}")

    for idx, z in enumerate(z_levels):
        # Use trimesh section which is robust and can distinguish loops
        try:
            section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        except Exception:
            continue

        if section is None or len(section.entities) == 0:
            continue

        # Transform 3D cut to 2D projection to find the outer loop
        planar_path, to_3D = section.to_2D()
        to_2D = np.linalg.inv(to_3D)
        local_centroid = trimesh.transform_points([mesh.centroid], to_2D)[0][:2]

        if len(planar_path.discrete) == 0:
            continue

        # Find the outermost loop by comparing the average radius of each loop
        outer_coords_2d = None
        max_avg_radius = -1.0
        for discrete_loop in planar_path.discrete:
            loop_points = np.array(discrete_loop)
            if len(loop_points) < 3:
                continue
            distances = np.linalg.norm(loop_points - local_centroid, axis=1)
            avg_radius = np.mean(distances)
            if avg_radius > max_avg_radius:
                max_avg_radius = avg_radius
                outer_coords_2d = loop_points

        if outer_coords_2d is None:
            continue

        # Project the 2D outer loop back to 3D
        z_filler = np.zeros(len(outer_coords_2d))
        layer_hits = trimesh.transform_points(np.column_stack((outer_coords_2d, z_filler)), to_3D)

        # --- Get Normals for the points on the outer loop ---
        # Create a ray query slightly outside the points, pointing inwards
        # This is a robust way to get the surface normal at a given XY location
        midpoints = (layer_hits[:-1] + layer_hits[1:]) / 2.0
        radial_vectors = midpoints[:, :2] - local_centroid[:2]
        radial_vectors_norm = radial_vectors / (np.linalg.norm(radial_vectors, axis=1, keepdims=True) + 1e-9)
        
        # Start rays from slightly outside the model, pointing toward the centroid
        ray_origins = midpoints.copy()
        ray_origins[:, :2] += radial_vectors_norm * model_scale * 0.1
        # Create 3D direction vectors [-dx, -dy, 0]
        ray_directions = np.zeros((len(radial_vectors_norm), 3))
        ray_directions[:, :2] = -radial_vectors_norm

        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        # Map the found normals back to the original points
        layer_normals = np.zeros_like(layer_hits)
        if len(index_ray) > 0:
            # Get the face normal for each triangle hit
            hit_normals = face_normals[index_tri]
            
            # Create a mapping from ray index to the first normal found for that ray
            # This handles cases where a ray hits multiple triangles
            unique_rays, first_indices = np.unique(index_ray, return_index=True)
            ray_to_normal_map = {r: n for r, n in zip(unique_rays, hit_normals[first_indices])}

            # Assign normals to the corresponding segment midpoint
            for i in range(len(midpoints)):
                if i in ray_to_normal_map:
                    # Assign the found normal to both points of the segment
                    layer_normals[i] = ray_to_normal_map[i]
                    layer_normals[i+1] = ray_to_normal_map[i]

            # If some normals are still zero, fill them with their non-zero neighbor
            for i in range(len(layer_normals)):
                if np.allclose(layer_normals[i], 0):
                    # Find the nearest non-zero normal
                    for j in range(1, len(layer_normals)):
                        prev_idx = (i - j + len(layer_normals)) % len(layer_normals)
                        next_idx = (i + j) % len(layer_normals)
                        if not np.allclose(layer_normals[prev_idx], 0):
                            layer_normals[i] = layer_normals[prev_idx]
                            break
                        if not np.allclose(layer_normals[next_idx], 0):
                            layer_normals[i] = layer_normals[next_idx]
                            break
        else:
            # Fallback: use a horizontal normal if ray casting fails
            print(f"Warning: Ray-based normal detection failed at Z={z:.2f}. Using fallback.")
            radial_fallback = layer_hits - mesh.centroid
            radial_fallback[:, 2] = 0 # Project to XY plane
            layer_normals = radial_fallback / (np.linalg.norm(radial_fallback, axis=1, keepdims=True) + 1e-9)
        
        if len(layer_hits) > 5:
            if last_layer_end_point is None:
                start_ref = layer_hits[0]
            else:
                start_ref = last_layer_end_point
            
            distances_to_start = np.linalg.norm(layer_hits - start_ref, axis=1)
            start_idx = np.argmin(distances_to_start)
            sorted_hits = np.roll(layer_hits, -start_idx, axis=0)
            
            if not forward_direction:
                sorted_hits = np.flip(sorted_hits, axis=0)
            
            forward_direction = not forward_direction
            last_layer_end_point = sorted_hits[-1]
            
            unique_mask = np.ones(len(sorted_hits), dtype=bool)
            for i in range(1, len(sorted_hits)):
                if np.allclose(sorted_hits[i], sorted_hits[i - 1], atol=1e-3):
                    unique_mask[i] = False
            filtered_hits = sorted_hits[unique_mask]
            
            # Filter normals to match filtered hits
            sorted_normals = np.roll(layer_normals, -start_idx, axis=0)
            if not forward_direction: # The original code had a bug here, it should be `if forward_direction` to match the flip
                sorted_normals = np.flip(sorted_normals, axis=0)
            filtered_normals = sorted_normals[unique_mask]
            
            sorted_hits_t = filtered_hits.T.astype(float)
            num_points_filtered = sorted_hits_t.shape[1]

            if num_points_filtered > 4:
                spline_degree = min(3, num_points_filtered - 1)
                is_periodic = np.allclose(sorted_hits_t[:, 0], sorted_hits_t[:, -1], atol=1e-2)
                
                if is_periodic:
                    sorted_hits_t = sorted_hits_t[:, :-1]
                
                try:
                    tck, u = splprep(
                        sorted_hits_t, 
                        s=smoothing, 
                        per=is_periodic, 
                        k=spline_degree
                    )
                    
                    u_fine = np.linspace(0, 1, 400)
                    smooth_xyz = np.array(splev(u_fine, tck)).T
                    
                    # Interpolate vector trajectory fields [I, J, K]
                    tck_n, u_n = splprep(filtered_normals.T, s=smoothing, per=True, k=3)
                    smooth_ijk = np.array(splev(u_fine, tck_n)).T
                    smooth_ijk /= np.linalg.norm(smooth_ijk, axis=1, keepdims=True) # Normalize vectors
                    
                    # Ensure path is closed if it was periodic
                    if np.allclose(filtered_hits[0], filtered_hits[-1], atol=1e-3) or tck[0][-1] == tck[0][0]:
                        if not np.allclose(smooth_xyz[0], smooth_xyz[-1]):
                            smooth_xyz = np.vstack([smooth_xyz, smooth_xyz[0]])
                            smooth_ijk = np.vstack([smooth_ijk, smooth_ijk[0]]) # Close normals too

                    green_paths.append(smooth_xyz)
                    green_points_data.append(smooth_xyz) # Store all smooth points
                    green_normals_data.append(smooth_ijk) # Store all smooth normals
                    
                except Exception as e:
                    print(f"Spline error at Z={z:.2f}: {e}. Rendering raw outer points fallback.")
                    # Re-add closure point for raw visual block fallback
                    closed_hits = np.vstack([filtered_hits, filtered_hits[0]])
                    closed_normals = np.vstack([filtered_normals, filtered_normals[0]])
                    
                    red_paths.append(closed_hits)
                    red_points_data.append(closed_hits)
                    red_normals_data.append(closed_normals) # Store fallback normals
            else:
                red_paths.append(filtered_hits)
                red_points_data.append(filtered_hits)
                red_normals_data.append(filtered_normals) # Store fallback normals

    # Create cylinder meshes for paths to give them thickness
    path_radius = line_thickness / 2.0
    if path_radius > 0:
        # Manually create a circular polygon to sweep along the path
        sections = 8
        theta = np.linspace(0, 2 * np.pi, sections, endpoint=False)
        vertices = np.column_stack([np.cos(theta), np.sin(theta)]) * path_radius
        circle_polygon = trimesh.path.polygons(vertices)

        if green_paths:
            green_meshes = []
            for p in green_paths:
                if len(p) > 1:
                    green_meshes.append(trimesh.path.sweep.sweep_polygon(circle_polygon, p))
            if green_meshes:
                path_mesh = trimesh.util.concatenate(green_meshes)
                path_mesh.visual.face_colors = [0, 255, 0, 255] # Green
                geometries.append(path_mesh)

        if red_paths:
            red_meshes = []
            for p in red_paths:
                if len(p) > 1:
                    red_meshes.append(trimesh.path.sweep.sweep_polygon(circle_polygon, p))
            if red_meshes:
                fallback_mesh = trimesh.util.concatenate(red_meshes)
                fallback_mesh.visual.face_colors = [255, 0, 0, 255] # Red
                geometries.append(fallback_mesh)

    return geometries, green_paths, red_paths, green_points_data, red_points_data, green_normals_data, red_normals_data


# --- Main Execution Logic ---

def main():
    mesh_path = "C:\\Users\\shish\\C-\\Slicing\\StepRobotFrames\\Adapter.stl"
    
    # Ensure the outputs directory exists
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize the visualizer with configuration
    visualizer = MeshVisualizer(
        mesh_path=mesh_path,
        wcs_origin=np.array([0, 0, 0]), # WCS axes at world origin
        stl_target_position=np.array([50, 50, 50]), # STL centroid at this absolute position (closer for better view)
        num_layers=150,
        sphere_radius=0.05,
        axis_length=10,
        path_thickness=0.01 # New: Control the thickness of the path lines
    )

    # 2. Execute the processing and visualization steps
    visualizer.generate_path_data()
    visualizer.visualize_points_as_spheres()
    visualizer.visualize_coordinate_frames()
    visualizer.generate_6dof_data()

    # 3. Get the final scene and show it
    final_scene = visualizer.get_scene()
    final_scene.show()

    # Optionally print some of the generated data
    if visualizer.toolpath_data:
        print("\nFirst 5 toolpath data entries:")
        for i in range(min(5, len(visualizer.toolpath_data))):
            print(visualizer.toolpath_data[i])

if __name__ == "__main__":
    main()
