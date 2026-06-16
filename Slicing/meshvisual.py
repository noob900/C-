import numpy as np
from scipy.interpolate import splprep, splev
import trimesh

def generate_trajectory_frames(mesh, num_points):
    trajectory_frames = []
    bounding_box = mesh.bounds
    res = max(2, int(np.sqrt(num_points / 3)))
    
    x_grid = np.linspace(bounding_box[0][0], bounding_box[1][0], res)
    y_grid = np.linspace(bounding_box[0][1], bounding_box[1][1], res)
    z_grid = np.linspace(bounding_box[0][2], bounding_box[1][2], res)

    scans = [
        (x_grid, y_grid, 2, bounding_box[1][2] + 5.0, [0, 0, -1]), # Top -> Down
        (y_grid, z_grid, 0, bounding_box[1][0] + 5.0, [-1, 0, 0]), # Side -> In
        (x_grid, z_grid, 1, bounding_box[1][1] + 5.0, [0, -1, 0])  # Front -> In
    ]

    for g1, g2, fixed_idx, fixed_val, direction in scans:
        for v1 in g1:
            for v2 in g2:
                origin = [0.0, 0.0, 0.0]
                if fixed_idx == 2: # XY plane
                    origin = [v1, v2, fixed_val]
                elif fixed_idx == 0: # YZ plane
                    origin = [fixed_val, v1, v2]
                else: # XZ plane
                    origin = [v1, fixed_val, v2]

                locations, index_ray, index_tri = mesh.ray.intersects_location(
                    ray_origins=np.array([origin]), 
                    ray_directions=np.array([direction])
                )

                if len(locations) > 0:
                    point = locations[0]
                    tri_index = index_tri[0]
                    z_axis = mesh.face_normals[tri_index].copy()
                    z_axis /= np.linalg.norm(z_axis)

                    x_axis = np.array([1.0, 0.0, 0.0])
                    if np.abs(np.dot(z_axis, x_axis)) > 0.95:
                        x_axis = np.array([0.0, 1.0, 0.0])

                    y_axis = np.cross(z_axis, x_axis)
                    y_axis /= np.linalg.norm(y_axis)

                    x_axis = np.cross(y_axis, z_axis)
                    x_axis /= np.linalg.norm(x_axis)

                    frame = np.identity(4)
                    frame[0:3, 0] = x_axis
                    frame[0:3, 1] = y_axis
                    frame[0:3, 2] = z_axis
                    frame[0:3, 3] = point

                    trajectory_frames.append(frame)

    return trajectory_frames


def generate_smooth_layered_path(mesh_path, num_layers=30, smoothing=0.1, line_thickness=0.05):
    """
    Generates a continuous layered toolpath isolating ONLY the outermost path.
    Uses pure numpy math arrays to segment inner vs outer tracks, bypasses shapely entirely.
    """
    mesh = trimesh.load(mesh_path, force='mesh')
    
    # Use solid gray. Transparency (alpha < 255) significantly slows down the 3D renderer.
    mesh.visual.face_colors = np.full((len(mesh.faces), 4), [180, 180, 180, 255], dtype=np.uint8)
    
    bounds = mesh.bounds
    centroid = mesh.centroid
    
    # Generate continuous Z increments from bottom to top
    z_levels = np.linspace(bounds[0][2], bounds[1][2], num_layers)
    
    geometries = [mesh]
    geometries = []
    last_layer_end_point = None
    green_paths = []
    red_paths = []
    green_points_data = [] # New: to store points for successful paths
    red_points_data = []   # New: to store points for fallback paths
    forward_direction = True

    print(f"Slicing model boundaries from Min Z={bounds[0][2]:.2f} to Max Z={bounds[1][2]:.2f}")

    for z in z_levels:
        try:
            # Cut the mesh flat at height 'z' using a horizontal slicing normal vector
            section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        except Exception:
            continue
            
        if section is None:
            continue
            
        # Transform the 3D cut into flat 2D projection lines
        planar_path, to_3D = section.to_2D()
        
        # Correctly project the global centroid into the local 2D space of the section.
        # This is necessary because to_planar() defines its own local 2D origin.
        to_2D = np.linalg.inv(to_3D)
        local_centroid = trimesh.transform_points([centroid], to_2D)[0][:2]
        
        # --- PURE NUMPY GEOMETRY EXTRACTION (No Shapely Required) ---
        if len(planar_path.discrete) == 0:
            continue
            
        # planar_path.discrete contains lists of coordinates for each loop.
        # We loop through them and identify which loop has the maximum average radius from the center.
        outer_coords_2d = None
        max_avg_radius = -1.0
        
        for discrete_loop in planar_path.discrete:
            loop_points = np.array(discrete_loop)
            if len(loop_points) < 3:
                continue
                
            # Compute average distance of this specific loop profile from the center line
            distances = np.linalg.norm(loop_points - local_centroid, axis=1)
            avg_radius = np.mean(distances)
            
            # The outer diameter path will always have a larger average radius than the inner diameter
            if avg_radius > max_avg_radius:
                max_avg_radius = avg_radius
                outer_coords_2d = loop_points

        if outer_coords_2d is None:
            continue
        
        # Project 2D perimeter vertices back into 3D space using Trimesh's native matrix transform
        z_filler = np.zeros(len(outer_coords_2d))
        layer_hits = trimesh.transform_points(np.column_stack((outer_coords_2d, z_filler)), to_3D)
        
        if len(layer_hits) > 5:
            if last_layer_end_point is None:
                start_ref = layer_hits[0]
            else:
                start_ref = last_layer_end_point
            
            # Roll the already ordered boundary loop coordinates to match the closest point from last layer
            distances_to_start = np.linalg.norm(layer_hits - start_ref, axis=1)
            start_idx = np.argmin(distances_to_start)
            sorted_hits = np.roll(layer_hits, -start_idx, axis=0)
            
            if not forward_direction:
                sorted_hits = np.flip(sorted_hits, axis=0)
            
            forward_direction = not forward_direction
            last_layer_end_point = sorted_hits[-1]
            
            # Filter identical matching sequential duplicate nodes
            unique_mask = np.ones(len(sorted_hits), dtype=bool)
            for i in range(1, len(sorted_hits)):
                if np.allclose(sorted_hits[i], sorted_hits[i - 1], atol=1e-3):
                    unique_mask[i] = False
            filtered_hits = sorted_hits[unique_mask]
            
            sorted_hits_t = filtered_hits.T.astype(float)
            num_points_filtered = sorted_hits_t.shape[1]
            
            if num_points_filtered > 4:
                spline_degree = min(3, num_points_filtered - 1)
                is_periodic = np.allclose(sorted_hits_t[:, 0], sorted_hits_t[:, -1], atol=1e-2)
                
                if is_periodic:
                    sorted_hits_t = sorted_hits_t[:, :-1]
                
                try:
                    # Execute math curve fitting calculation
                    tck, u = splprep(
                        sorted_hits_t, 
                        s=smoothing, 
                        per=is_periodic, 
                        k=spline_degree
                    )
                    
                    u_fine = np.linspace(0, 1, 400)
                    smooth_points = np.array(splev(u_fine, tck)).T
                    
                    if is_periodic or not np.allclose(smooth_points[0], smooth_points[-1]):
                        smooth_points = np.vstack([smooth_points, smooth_points[0]])
                    
                    # Store points to bundle them later for faster rendering
                    green_paths.append(smooth_points)
                    # Add the original filtered_hits as points for visualization
                    green_points_data.append(filtered_hits)
                    
                except Exception as e:
                    print(f"Spline error at Z={z:.2f}: {e}. Rendering raw outer points fallback.")
                    red_paths.append(filtered_hits)
                    # Add the filtered_hits as points for visualization
                    red_points_data.append(filtered_hits)
            else:
                red_paths.append(filtered_hits)
                # Add the filtered_hits as points for visualization
                red_points_data.append(filtered_hits)

    # Efficiently bundle paths into single objects to speed up visualization
    if green_paths:
        # Loading a list of paths creates one Path3D object with multiple entities
        path_obj = trimesh.util.concatenate([trimesh.load_path(p) for p in green_paths])
        path_obj.colors = np.full((len(path_obj.entities), 4), [0, 255, 0, 255], dtype=np.uint8)
        geometries.append(path_obj)

    if red_paths:
        fallback_obj = trimesh.util.concatenate([trimesh.load_path(p) for p in red_paths])
        fallback_obj.colors = np.full((len(fallback_obj.entities), 4), [255, 0, 0, 255], dtype=np.uint8)
        geometries.append(fallback_obj)

    # Add points visualization
    if green_points_data:
        all_green_points = np.vstack(green_points_data)
        green_point_cloud = trimesh.points.PointCloud(all_green_points)
        green_point_cloud.colors = [0, 0, 0, 255] 
        geometries.append(green_point_cloud)

    if red_points_data:
        all_red_points = np.vstack(red_points_data)
        red_point_cloud = trimesh.points.PointCloud(all_red_points)
        red_point_cloud.colors = [255, 0, 0, 255] # Same color as paths
        geometries.append(red_point_cloud)

    
    return geometries, green_paths, red_paths, green_points_data, red_points_data
    



def visualize_spheres_at_points(mesh_path, sphere_radius=1.0):
    
    # 1. Generate trajectory points using the file path
    # trajectory returns (geometries, green_paths, red_paths, green_points_data, red_points_data)
    # We reduce num_layers for performance when rendering thousands of individual spheres
    trajectory = generate_smooth_layered_path(mesh_path, num_layers=30, smoothing=0.1, line_thickness=0.04)
    green_points_list = trajectory[3] # Extract the green_points_data

    if not green_points_list:
        print("No points were generated for visualization.")
        return

    # 2. Flatten the list of arrays into a single array of points
    all_points = np.vstack(green_points_list)
    
    # 3. Create spheres efficiently as a single mesh using NumPy broadcasting
    # This is much faster than creating thousands of separate Trimesh objects in a loop.
    # We use a low-poly icosphere (subdivisions=1) for faster rendering.
    proto = trimesh.creation.icosphere(subdivisions=1, radius=sphere_radius)
    v_proto, f_proto = proto.vertices, proto.faces
    n_v, n_f, n_p = len(v_proto), len(f_proto), len(all_points)

    # Broadcast the prototype sphere vertices to all point locations
    all_v = np.tile(v_proto, (n_p, 1)) + np.repeat(all_points, n_v, axis=0)
    
    # Tile the faces and apply the vertex index offsets
    offsets = np.arange(n_p) * n_v
    all_f = np.tile(f_proto, (n_p, 1)) + np.repeat(offsets, n_f)[:, None]

    combined_spheres = trimesh.Trimesh(vertices=all_v, faces=all_f, process=False)
    combined_spheres.visual.face_colors = [0, 255, 0, 255]
        
    # 4. Show only the spheres
    trimesh.Scene(combined_spheres).show()

if __name__ == "__main__":
    mesh_path = "C:\\Users\\shish\\C-\\Slicing\\StepRobotFrames\\Adapter.stl"
    
    # Execute the library-independent geometry analyzer
    #generate_smooth_layered_path(mesh_path, num_layers=100, smoothing=0.1, line_thickness=0.04)
    visualize_spheres_at_points(mesh_path, sphere_radius=0.1)