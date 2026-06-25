import numpy as np
from scipy.interpolate import splprep, splev
import trimesh
from trimesh.path import Path3D
from typing import Tuple, List, Dict, Union
import math # Already imported, but good to keep
import os
import pyvista as pv # Added for 3D visualization and interaction

# --- PyVista Helper Functions (Moved from being implicitly called) ---
def _trimesh_to_pyvista_mesh(trimesh_mesh: trimesh.Trimesh) -> pv.PolyData:
    """Converts a trimesh.Trimesh object to a pyvista.PolyData object."""
    if trimesh_mesh.is_empty:
        return pv.PolyData()
    
    # PyVista expects faces as (n_points_in_face, p1_idx, p2_idx, ...)
    # Trimesh stores faces as (p1_idx, p2_idx, p3_idx)
    faces = np.hstack((np.full((len(trimesh_mesh.faces), 1), 3), trimesh_mesh.faces))
    pv_mesh = pv.PolyData(trimesh_mesh.vertices, faces)
    
    return pv_mesh

def _trimesh_path3d_to_pyvista_lines(trimesh_path: Path3D) -> pv.PolyData:
    """Converts a trimesh.Path3D object to a pyvista.PolyData object for lines."""
    if trimesh_path is None or len(trimesh_path.vertices) == 0:
        return pv.PolyData()

    # The 'lines' array for PyVista needs to be in the format:
    # [n_points_1, p1_idx, p2_idx, ..., n_points_2, p1_idx, p2_idx, ...]
    # For a Path3D, each entity is a line segment.
    lines_list = []
    for entity in trimesh_path.entities:
        # entity.points contains the vertex indices for this line segment
        lines_list.append(len(entity.points))
        lines_list.extend(entity.points)
        
    return pv.PolyData(trimesh_path.vertices, lines=np.array(lines_list))

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

def _create_batch_point_frames(points, normals=None, axis_length=2.0):
    """
    Creates small RGB coordinate frames at each provided point efficiently.
    If normals are provided, the Z-axis of each frame is aligned with the surface normal.
    
    Args:
        points: Nx3 array of point positions
        normals: Nx3 array of surface normals (optional). If None, uses world-aligned frames.
        axis_length: Length of the axis lines
        
    Returns:
        List of Path3D objects for [x_path, y_path, z_path]
    """
    if len(points) == 0:
        return []
    
    # Check if normals are provided and valid
    use_normals = normals is not None and len(normals) == len(points)
    
    # Create line segments for each axis
    x_segs = []
    y_segs = []
    z_segs = []
    
    for i, p in enumerate(points):
        if use_normals:
            # Get the normal and compute orthogonal frame
            normal = normals[i]
            normal = normal / (np.linalg.norm(normal) + 1e-9)  # Normalize
            
            # Use the _normal_to_rpy function logic to create frame axes
            # Z-axis is aligned with the normal
            z_axis = normal
            
            # Pick a reference axis that's not parallel to the normal
            ref = np.array([1.0, 0.0, 0.0])
            if np.abs(np.dot(ref, z_axis)) > 0.95:
                ref = np.array([0.0, 1.0, 0.0])
            if np.abs(np.dot(ref, z_axis)) > 0.95:
                ref = np.array([0.0, 0.0, 1.0])
            
            # Compute X-axis orthogonal to Z
            x_axis = ref - z_axis * np.dot(ref, z_axis)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
            
            # Compute Y-axis orthogonal to both X and Z
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
        else:
            # World-aligned axes
            x_axis = np.array([1.0, 0.0, 0.0])
            y_axis = np.array([0.0, 1.0, 0.0])
            z_axis = np.array([0.0, 0.0, 1.0])
        
        # Create axis endpoints relative to point
        x_segs.append([p, p + x_axis * axis_length])
        y_segs.append([p, p + y_axis * axis_length])
        z_segs.append([p, p + z_axis * axis_length])
        
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
        self.mesh: trimesh.Trimesh = None # type: ignore
        
        # Data generated by processing
        self.points: np.ndarray = np.empty((0, 3))
        self.normals: np.ndarray = np.empty((0, 3))
        self.green_paths: List[np.ndarray] = []
        self.red_paths: List[np.ndarray] = []
        self.toolpath_data: List[Dict] = []

        # Geometries for visualization
        self._wcs_frames_trimesh: List[Union[trimesh.Trimesh, Path3D]] = []
        self._stl_frames_trimesh: List[Union[trimesh.Trimesh, Path3D]] = []
        self._sampled_point_frames_trimesh: List[Union[trimesh.Trimesh, Path3D]] = []

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

        _, self.green_paths, self.red_paths, green_points_data, red_points_data, green_normals_data, red_normals_data = generate_smooth_layered_path_from_mesh(
            self.mesh, num_layers=self.num_layers, smoothing=smoothing, line_thickness=self.path_thickness
        )

        # Consolidate all generated points and normals
        all_points_parts = []
        if green_points_data:
            all_points_parts.append(np.vstack(green_points_data))
        if red_points_data:
            all_points_parts.append(np.vstack(red_points_data))
        
        all_normals_parts = []
        if green_normals_data:
            all_normals_parts.append(np.vstack(green_normals_data))
        if red_normals_data:
            all_normals_parts.append(np.vstack(red_normals_data))

        if all_points_parts:
            self.points = np.vstack(all_points_parts)
        if all_normals_parts:
            self.normals = np.vstack(all_normals_parts)
        print(f"Generated {len(self.points)} points and {len(self.normals)} normals.")

    # Removed visualize_points_as_spheres as PyVista will handle point visualization and picking directly.

    def visualize_coordinate_frames(self):
        """
        Generates WCS, STL origin, and a sample of point frames aligned with surface normals.
        These are stored as trimesh objects for use in get_scene().
        """
        # WCS visualization
        self._wcs_frames_trimesh = list(_create_wcs_origin(self.wcs_origin, self.axis_length))

        # STL reference point visualization (magenta axes)
        stl_x, stl_y, stl_z, stl_sphere = _create_wcs_origin(self.stl_target_position, self.axis_length * 0.5)
        stl_x.colors = np.array([[255, 0, 255, 255]], dtype=np.uint8)
        stl_y.colors = np.array([[255, 0, 255, 255]], dtype=np.uint8)
        stl_z.colors = np.array([[255, 0, 255, 255]], dtype=np.uint8)
        stl_sphere.visual.vertex_colors = [255, 0, 255, 255] # type: ignore
        self._stl_frames_trimesh = [stl_x, stl_y, stl_z, stl_sphere]

        # Sampled point frames aligned with surface normals
        if len(self.points) > 0 and len(self.normals) > 0:
            num_frames = min(100, len(self.points))
            sample_indices = np.linspace(0, len(self.points) - 1, num_frames, dtype=int)
            if len(sample_indices) > 0:
                sampled_points = self.points[sample_indices]
                sampled_normals = self.normals[sample_indices]
                self._sampled_point_frames_trimesh = _create_batch_point_frames(
                    sampled_points, normals=sampled_normals, axis_length=self.axis_length * 0.2
                )
        
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

    def print_6dof_data_for_index(self, index: int):
        """
        Prints the XYZ coordinates and ABC (Euler) angles for a specific toolpath point.
        This simulates the "display in small text bottom right" feature by printing to console.
        """
        if not self.toolpath_data:
            print("No 6-DOF toolpath data available. Please run generate_6dof_data first.")
            return
        if not (0 <= index < len(self.toolpath_data)):
            print(f"Index {index} is out of bounds. Available range: 0 to {len(self.toolpath_data) - 1}.")
            return
        
        data = self.toolpath_data[index]
        print(f"\n--- Data for Point Index {index} (relative to WCS) ---")
        print(f"Coordinates (XYZ): X={data['X']:.3f}, Y={data['Y']:.3f}, Z={data['Z']:.3f}")
        print(f"Euler Angles (ABC - Roll, Pitch, Yaw): A={data['A']:.2f}°, B={data['B']:.2f}°, C={data['C']:.2f}°")
        print("--------------------------------------------------")

    def get_scene(self) -> pv.Plotter:
        """
        Returns a PyVista Plotter object with all generated geometries and interactive picking.
        """
        plotter = pv.Plotter()

        # Add the main mesh
        pv_mesh = _trimesh_to_pyvista_mesh(self.mesh)
        plotter.add_mesh(pv_mesh, color='lightgray', show_edges=False)
        
        # Add WCS frames
        for geom in self._wcs_frames_trimesh:
            if isinstance(geom, Path3D):
                plotter.add_mesh(_trimesh_path3d_to_pyvista_lines(geom), line_width=3)
            elif isinstance(geom, trimesh.Trimesh): # Sphere
                plotter.add_mesh(_trimesh_to_pyvista_mesh(geom), color=geom.visual.vertex_colors[0][:3]/255.0)
        
        # Add STL frames
        for geom in self._stl_frames_trimesh:
            if isinstance(geom, Path3D):
                plotter.add_mesh(_trimesh_path3d_to_pyvista_lines(geom), line_width=2)
            elif isinstance(geom, trimesh.Trimesh): # Sphere
                plotter.add_mesh(_trimesh_to_pyvista_mesh(geom), color=geom.visual.vertex_colors[0][:3]/255.0)

        # Add generated toolpaths (green and red lines) as PyVista lines
        for path_points in self.green_paths:
            if len(path_points) > 1:
                plotter.add_lines(path_points, color='green', width=2, connected=True)
        for path_points in self.red_paths:
            if len(path_points) > 1:
                plotter.add_lines(path_points, color='red', width=2, connected=True)

        # Add sampled point frames
        for geom in self._sampled_point_frames_trimesh: # This list contains Path3D objects
            if isinstance(geom, Path3D):
                plotter.add_mesh(_trimesh_path3d_to_pyvista_lines(geom), line_width=1)

        # Create a PolyData object for the toolpath points and attach 6-DOF data for picking
        if len(self.points) > 0 and self.toolpath_data:
            toolpath_pv_points = pv.PolyData(self.points)
            
            # Attach 6-DOF data as point data
            # Convert list of dicts to dict of arrays
            data_arrays = {key: np.array([d[key] for d in self.toolpath_data]) for key in self.toolpath_data[0].keys()}
            for key, arr in data_arrays.items():
                toolpath_pv_points.point_data[key] = arr

            # Add the points to the plotter. Render as spheres for better visibility and picking.
            # point_size is in screen pixels.
            plotter.add_mesh(toolpath_pv_points, color='black', point_size=10, render_points_as_spheres=True, name='toolpath_points', pickable=True)

            # Setup picking callback
            def pick_callback(): # Callback now takes no arguments
                # First, clear any existing text
                if plotter.actors.get('picked_info_text'):
                    plotter.remove_actor('picked_info_text', render=False)

                # If a point is picked, find it and display its data
                if plotter.picked_point is not None:
                    point_id = toolpath_pv_points.find_closest_point(plotter.picked_point)
                    
                    if point_id >= 0:
                        # Retrieve the data for the found point ID
                        x = toolpath_pv_points.point_data['X'][point_id]
                        y = toolpath_pv_points.point_data['Y'][point_id]
                        z = toolpath_pv_points.point_data['Z'][point_id]
                        a = toolpath_pv_points.point_data['A'][point_id]
                        b = toolpath_pv_points.point_data['B'][point_id]
                        c = toolpath_pv_points.point_data['C'][point_id]

                        # Add new text
                        plotter.add_text(
                            f"XYZ: ({x:.3f}, {y:.3f}, {z:.3f})\nABC: ({a:.2f}°, {b:.2f}°, {c:.2f}°)",
                            position='lower_right', font_size=10, color='white', name='picked_info_text'
                        )

            plotter.enable_point_picking(callback=pick_callback, show_point=True, color='yellow', point_size=15)
            print("Interactive point picking enabled.")

        return plotter

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
    
    last_layer_end_point = None
    green_paths = [] # List of numpy arrays, each representing a path
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
        # Calculate centroid in 2D for finding outermost loop
        if len(planar_path.discrete) > 0:
            all_points_2d = np.vstack([loop for loop in planar_path.discrete if len(loop) > 0])
            if len(all_points_2d) > 0:
                local_centroid = np.mean(all_points_2d, axis=0)
            else:
                local_centroid = np.array([0,0]) # Fallback

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
        # To get normals for each point, we can cast rays from slightly offset points
        # along the local radial direction (from the 2D centroid of the slice)
        
        # Calculate 3D centroid of the current layer_hits
        layer_centroid_3d = np.mean(layer_hits, axis=0)
        
        ray_origins = []
        ray_directions = []
        
        # For each point on the contour, cast a ray from slightly outside, pointing inwards
        for p_idx, p in enumerate(layer_hits):
            # Vector from layer centroid to point p, projected to XY plane
            radial_vec_xy = p[:2] - layer_centroid_3d[:2]
            radial_vec_xy_norm = radial_vec_xy / (np.linalg.norm(radial_vec_xy) + 1e-9)
            
            # Start ray slightly outside the point along the radial direction
            origin = p.copy()
            origin[:2] += radial_vec_xy_norm * model_scale * 0.01 # Small offset
            
            # Direction is inwards, towards the centroid
            direction = -radial_vec_xy_norm
            ray_origins.append(origin)
            ray_directions.append(np.array([direction[0], direction[1], 0.0])) # Keep Z direction 0 for horizontal rays

        ray_origins = np.array(ray_origins)
        ray_directions = np.array(ray_directions)

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
            ray_to_normal_map = {r: n for r, n in zip(unique_rays, hit_normals[first_indices])} # Map ray index to normal

            # Assign normals to the corresponding point
            for i in range(len(layer_hits)):
                if i in ray_to_normal_map:
                    layer_normals[i] = ray_to_normal_map[i] # Assign the found normal
                else:
                    # If a ray didn't hit, try to interpolate from neighbors or use a fallback
                    # This is a simple fallback, more robust methods might involve nearest neighbor search
                    pass # Will be handled by the next loop if still zero

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
            # Fallback: if no rays hit at all, use a simple normal (e.g., Z-up or radial)
            print(f"Warning: Ray-based normal detection failed for all points at Z={z:.2f}. Using fallback.")
            radial_fallback = layer_hits - mesh.centroid
            radial_fallback[:, 2] = 0 # Project to XY plane
            layer_normals = radial_fallback / (np.linalg.norm(radial_fallback, axis=1, keepdims=True) + 1e-9)
            # If normalization resulted in NaNs (from a zero-length vector), replace the entire vector.
            nan_rows = np.any(np.isnan(layer_normals), axis=1)
            layer_normals[nan_rows] = [0, 0, 1]
        # Ensure all normals are normalized
        layer_normals = layer_normals / (np.linalg.norm(layer_normals, axis=1, keepdims=True) + 1e-9)
        
        if len(layer_hits) > 1: # Need at least 2 points to form a path
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
                if np.allclose(sorted_hits[i], sorted_hits[i - 1], atol=1e-4): # Increased precision for duplicate check
                    unique_mask[i] = False
            filtered_hits = sorted_hits[unique_mask]
            
            # Filter normals to match filtered hits
            sorted_normals = np.roll(layer_normals, -start_idx, axis=0)
            if not forward_direction:
                sorted_normals = np.flip(sorted_normals, axis=0)
            filtered_normals = sorted_normals[unique_mask]
            
            num_points_filtered = len(filtered_hits)

            if num_points_filtered > 1: # Need at least 2 points for splprep
                spline_degree = min(3, num_points_filtered - 1)
                is_periodic = np.allclose(filtered_hits[0], filtered_hits[-1], atol=1e-2)
                
                # For periodic splines, splprep expects the first and last point to be distinct
                # If they are identical, remove the last one and set per=True
                if is_periodic and num_points_filtered > 1 and np.allclose(filtered_hits[0], filtered_hits[-1]):
                    filtered_hits_for_spline = filtered_hits[:-1]
                    filtered_normals_for_spline = filtered_normals[:-1]
                    num_points_for_spline = len(filtered_hits_for_spline)
                    spline_degree = min(3, num_points_for_spline - 1)
                    if num_points_for_spline < 4 and is_periodic: # Not enough points for cubic periodic spline
                        is_periodic = False # Cannot make periodic spline with too few points
                else:
                    filtered_hits_for_spline = filtered_hits
                    filtered_normals_for_spline = filtered_normals
                    num_points_for_spline = len(filtered_hits_for_spline)
                    spline_degree = min(3, num_points_for_spline - 1)
                    is_periodic = False # Explicitly set to False if not periodic or not enough points

                if num_points_for_spline > 1: # Need at least 2 points for splprep
                    try:
                        u_fine = np.linspace(0, 1, 400)

                        # Interpolate 3D location paths [X, Y, Z]
                        tck_pos, u_pos = splprep(
                            filtered_hits_for_spline.T, 
                            s=smoothing, 
                            per=is_periodic, 
                            k=spline_degree
                        )
                        smooth_xyz = np.array(splev(u_fine, tck_pos)).T
                        
                        # Interpolate vector trajectory fields [I, J, K] (normals)
                        tck_n, u_n = splprep(
                            filtered_normals_for_spline.T, 
                            s=smoothing, 
                            per=is_periodic, # Use same periodicity for normals
                            k=spline_degree
                        )
                        smooth_ijk = np.array(splev(u_fine, tck_n)).T
                        smooth_ijk /= (np.linalg.norm(smooth_ijk, axis=1, keepdims=True) + 1e-9) # Normalize vectors
                        
                        # Ensure path is closed if it was periodic
                        if is_periodic and not np.allclose(smooth_xyz[0], smooth_xyz[-1]):
                                smooth_xyz = np.vstack([smooth_xyz, smooth_xyz[0]])
                                smooth_ijk = np.vstack([smooth_ijk, smooth_ijk[0]]) # Close normals too

                        green_paths.append(smooth_xyz)
                        green_points_data.append(smooth_xyz) # Store all smooth points
                        green_normals_data.append(smooth_ijk) # Store all smooth normals
                    except Exception as e:
                        print(f"Spline error at Z={z:.2f}: {e}. Rendering raw outer points fallback.")
                        # Fallback to raw points if spline generation fails
                        closed_hits = np.vstack([filtered_hits, filtered_hits[0]]) if is_periodic else filtered_hits
                        closed_normals = np.vstack([filtered_normals, filtered_normals[0]]) if is_periodic else filtered_normals
                        
                        red_paths.append(closed_hits)
                        red_points_data.append(closed_hits)
                        red_normals_data.append(closed_normals) # Store fallback normals
            else: # Not enough points for splprep (num_points_filtered < 2)
                print(f"Warning: Not enough points ({num_points_filtered}) for path at Z={z:.2f}. Falling back to raw points.")
                closed_hits = np.vstack([filtered_hits, filtered_hits[0]]) if is_periodic else filtered_hits
                closed_normals = np.vstack([filtered_normals, filtered_normals[0]]) if is_periodic else filtered_normals
                # If there's only one point, closed_hits might still be just one point.
                # Ensure it's not empty before appending.
                if len(closed_hits) > 0:
                    red_paths.append(closed_hits)
                    red_points_data.append(closed_hits)
                red_normals_data.append(filtered_normals) # Store fallback normals

    return [], green_paths, red_paths, green_points_data, red_points_data, green_normals_data, red_normals_data


# --- Main Execution Logic ---

def main():
    mesh_path = r"C:\Users\shish\C-\Slicing\15778_NoveltyBust_EgyptianPharaoh_V1_NEW.obj"
    
    # Ensure the outputs directory exists
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize the visualizer with configuration
    visualizer = MeshVisualizer(
        mesh_path=mesh_path, 
        wcs_origin=np.array([0, 0, 0]), # WCS axes at world origin
        stl_target_position=np.array([0, 0, 0]), # STL centroid at this absolute position (closer for better view)
        num_layers=150,
        sphere_radius=0.02,
        axis_length=10,
        path_thickness=0.001 # New: Control the thickness of the path lines
    )

    # 2. Execute the processing and visualization steps
    visualizer.generate_path_data() # This now populates points and normals
    # Removed visualize_points_as_spheres as PyVista handles point visualization and picking directly.
    visualizer.visualize_coordinate_frames() # Added to show WCS and sampled point frames
    visualizer.generate_6dof_data() # Generate the XYZABC data

    # 3. Print 6-DOF data for a sample point
    if visualizer.toolpath_data:
        # Choose a point to display, e.g., the first point or a point in the middle
        sample_index = len(visualizer.toolpath_data) // 2
        visualizer.print_6dof_data_for_index(sample_index)
    else:
        print("No toolpath data generated to display sample 6-DOF information.")

    # 4. Get the final scene and show it
    final_scene = visualizer.get_scene()
    # Also try to show if viewer is available
    try:
        final_scene.show()
    except Exception as e:
        print(f"Viewer not available ({e}), but scene has been saved.")

if __name__ == "__main__":
    main()
