import numpy as np
from scipy.interpolate import splprep, splev
import trimesh
def generate_5axis_cl_path(mesh_path, num_layers=25, points_per_layer=100, smoothing=0.1):
    """
    Robust 5-axis toolpath engine using proximity routing.
    Fixed to eliminate Scipy 'Invalid inputs' and Trimesh 'line_to_cylinders' errors.
    """
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh.visual.face_colors = [180, 180, 180, 255] 
    
    bounds = mesh.bounds
    extents = mesh.extents
    model_scale = np.max(extents)
    
    line_thickness = model_scale * 0.005 
    vector_length = model_scale * 0.12
    
    print("--- 5-AXIS GEOMETRY ENGINE INITIALIZED ---")
    print(f"Slicing model layers between Z={bounds[0][2]:.2f} and Z={bounds[1][2]:.2f}")
    
    z_levels = np.linspace(bounds[0][2] + (extents[2] * 0.02), bounds[1][2] - (extents[2] * 0.02), num_layers)
    
    geometries = [mesh]
    five_axis_cl_data = []
    green_paths = []
    vector_paths = []
    fallback_paths = []
    green_points_data = []   # New: to store points for successful paths
    fallback_points_data = [] # New: to store points for fallback paths

    triangles = mesh.triangles
    face_normals = mesh.face_normals

    for idx, z in enumerate(z_levels):
        # 1. Linear Cross-Section Slicing
        z_min = np.min(triangles[:, :, 2], axis=1)
        z_max = np.max(triangles[:, :, 2], axis=1)
        intersect_mask = (z_min <= z) & (z_max >= z)
        
        intersecting_tris = triangles[intersect_mask]
        intersecting_normals = face_normals[intersect_mask]
        
        segments = []
        seg_normals = []
        
        for tri, normal in zip(intersecting_tris, intersecting_normals):
            pts = []
            for i in range(3):
                p1, p2 = tri[i], tri[(i + 1) % 3]
                if (p1[2] <= z <= p2[2]) or (p2[2] <= z <= p1[2]):
                    if not np.isclose(p1[2], p2[2]):
                        t = (z - p1[2]) / (p2[2] - p1[2])
                        pts.append(p1 + t * (p2 - p1))
            
            if len(pts) >= 2:
                segments.append((pts[0], pts[1]))
                seg_normals.append(normal)
                
        if len(segments) < 4:
            continue
            
        # 2. Proximity Chain Routing
        ordered_points = [segments[0][0], segments[0][1]]
        ordered_normals = [seg_normals[0], seg_normals[0]]
        used_segments = {0}
        
        for _ in range(len(segments)):
            current_tip = ordered_points[-1]
            best_dist = float('inf')
            best_idx = -1
            reverse_needed = False
            
            for s_idx, seg in enumerate(segments):
                if s_idx in used_segments:
                    continue
                
                d_start = np.linalg.norm(current_tip - seg[0])
                d_end = np.linalg.norm(current_tip - seg[1])
                
                if d_start < best_dist:
                    best_dist = d_start
                    best_idx = s_idx
                    reverse_needed = False
                if d_end < best_dist:
                    best_dist = d_end
                    best_idx = s_idx
                    reverse_needed = True
                    
            if best_idx != -1 and best_dist < (model_scale * 0.05):
                used_segments.add(best_idx)
                next_seg = segments[best_idx]
                if reverse_needed:
                    ordered_points.append(next_seg[0])
                else:
                    ordered_points.append(next_seg[1])
                ordered_normals.append(seg_normals[best_idx])
            else:
                break
                
        if len(ordered_points) < 5:
            continue
            
        layer_hits = np.array(ordered_points)
        layer_normals = np.array(ordered_normals)
        
        # 3. Clean Duplicate Vertices *BEFORE* Closing Loop (Prevents SciPy crashing)
        unique_mask = np.ones(len(layer_hits), dtype=bool)
        for i in range(1, len(layer_hits)):
            if np.allclose(layer_hits[i], layer_hits[i - 1], atol=1e-4):
                unique_mask[i] = False
                
        filtered_hits = layer_hits[unique_mask]
        filtered_normals = layer_normals[unique_mask]
        
        # Check if endpoints match; if they do, drop the last one for splprep(per=True)
        if len(filtered_hits) > 4 and np.allclose(filtered_hits[0], filtered_hits[-1], atol=1e-3):
            filtered_hits = filtered_hits[:-1]
            filtered_normals = filtered_normals[:-1]
        
        if len(filtered_hits) > 4:
            try:
                # Interpolate 3D location paths [X, Y, Z]
                tck_pos, u_pos = splprep(filtered_hits.T, s=smoothing, per=True, k=3)
                u_fine = np.linspace(0, 1, points_per_layer)
                smooth_xyz = np.array(splev(u_fine, tck_pos)).T
                
                # Interpolate vector trajectory fields [I, J, K]
                tck_n, u_n = splprep(filtered_normals.T, s=smoothing, per=True, k=3)
                smooth_ijk = np.array(splev(u_fine, tck_n)).T
                smooth_ijk /= np.linalg.norm(smooth_ijk, axis=1, keepdims=True)
                
                # Log finalized 5-axis entries
                for pos, vec in zip(smooth_xyz, smooth_ijk):
                    five_axis_cl_data.append([pos[0], pos[1], pos[2], vec[0], vec[1], vec[2]])
                
                # Aggregate paths for efficient batch rendering instead of creating individual tubes
                green_paths.append(smooth_xyz)
                # Add sampled points from the smooth path for visualization
                green_points_data.append(smooth_xyz[::10]) # Sample every 10th point

                for p, v in zip(smooth_xyz[::10], smooth_ijk[::10]):
                    vector_paths.append([p, p + v * vector_length])
                    
            except Exception as e:
                # Re-add closure point for raw visual block fallback
                closed_hits = np.vstack([filtered_hits, filtered_hits[0]])
                closed_normals = np.vstack([filtered_normals, filtered_normals[0]])
                
                for pos, vec in zip(closed_hits, closed_normals):
                    five_axis_cl_data.append([pos[0], pos[1], pos[2], vec[0], vec[1], vec[2]])
                    
                fallback_paths.append(closed_hits)
                # Add the closed_hits as points for visualization
                fallback_points_data.append(closed_hits)

    # Batch create Path3D objects to maximize FPS in the visualizer
    if green_paths:
        path_obj = trimesh.util.concatenate([trimesh.load_path(p) for p in green_paths])
        path_obj.colors = np.full((len(path_obj.entities), 4), [0, 255, 0, 255], dtype=np.uint8)
        geometries.append(path_obj)
        
    if vector_paths:
        vec_obj = trimesh.util.concatenate([trimesh.load_path(p) for p in vector_paths])
        vec_obj.colors = np.full((len(vec_obj.entities), 4), [0, 180, 255, 255], dtype=np.uint8)
        geometries.append(vec_obj)

    if fallback_paths:
        fail_obj = trimesh.util.concatenate([trimesh.load_path(p) for p in fallback_paths])
        fail_obj.colors = np.full((len(fail_obj.entities), 4), [255, 120, 0, 255], dtype=np.uint8)
        geometries.append(fail_obj)

    # Add points visualization
    if green_points_data:
        all_green_points = np.vstack(green_points_data)
        green_point_cloud = trimesh.points.PointCloud(all_green_points)
        green_point_cloud.colors = [0, 255, 0, 255] # Same color as paths
        geometries.append(green_point_cloud)

    if fallback_points_data:
        all_fallback_points = np.vstack(fallback_points_data)
        fallback_point_cloud = trimesh.points.PointCloud(all_fallback_points)
        fallback_point_cloud.colors = [255, 120, 0, 255] # Same color as paths
        geometries.append(fallback_point_cloud)

    cl_array = np.array(five_axis_cl_data)

    # 4. 3D Milling Cutter Position Simulation
    if len(cl_array) > 0:
        sample_index = len(cl_array) // 2
        tool_pos = cl_array[sample_index, 0:3]
        tool_dir = cl_array[sample_index, 3:6]

        z_axis = tool_dir
        x_axis = np.array([1, 0, 0]) if np.abs(tool_dir[0]) < 0.9 else np.array([0, 1, 0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        
        transform_matrix = np.identity(4)
        transform_matrix[0:3, 0] = x_axis
        transform_matrix[0:3, 1] = y_axis
        transform_matrix[0:3, 2] = z_axis
        transform_matrix[0:3, 3] = tool_pos

        tool_cone = trimesh.creation.cone(radius=model_scale * 0.025, height=model_scale * 0.12, transform=transform_matrix)
        tool_cone.visual.face_colors = [255, 0, 80, 240] 
        geometries.append(tool_cone)
        print(f"Generated a stable 5-axis trajectory containing {len(cl_array)} tool positions.")
    else:
        print("ERROR: No continuous path loops could be extracted from this mesh geometry.")

    print(f"Opening visualization pipeline containing {len(geometries)} components...")
    trimesh.Scene(geometries).show()
    return cl_array

if __name__ == "__main__":
    mesh_path = "C:\\Users\\shish\\C-\\Slicing\\StepRobotFrames\\Adapter.stl"
    cl_points = generate_5axis_cl_path(mesh_path, num_layers=25, points_per_layer=100)