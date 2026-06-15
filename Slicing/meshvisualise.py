import numpy as np
import trimesh

def generate_trajectory_frames(mesh, num_points):
    """
    Generates robot tool frames by casting rays against the mesh surface.
    Note: requires 'rtree' or 'pyembree' to be installed.
    """
    # 2. Initialize your empty list for the trajectory frames
    trajectory_frames = []

    # 3. Define the scanning resolution
    bounding_box = mesh.bounds
    # We divide by 3 because we are performing 3 orthogonal scans
    res = max(2, int(np.sqrt(num_points / 3)))
    
    # Create grids for all three axes
    x_grid = np.linspace(bounding_box[0][0], bounding_box[1][0], res)
    y_grid = np.linspace(bounding_box[0][1], bounding_box[1][1], res)
    z_grid = np.linspace(bounding_box[0][2], bounding_box[1][2], res)

    # 4. Define 3 orthogonal scan configurations: (Grid1, Grid2, Fixed_Axis_Index, Fixed_Value, Direction)
    # This ensures we scan from the Top, the Side, and the Front.
    scans = [
        (x_grid, y_grid, 2, bounding_box[1][2] + 5.0, [0, 0, -1]), # Top -> Down
        (y_grid, z_grid, 0, bounding_box[1][0] + 5.0, [-1, 0, 0]), # Side -> In
        (x_grid, z_grid, 1, bounding_box[1][1] + 5.0, [0, -1, 0])  # Front -> In
    ]

    for g1, g2, fixed_idx, fixed_val, direction in scans:
        for v1 in g1:
            for v2 in g2:
                # Build the ray origin based on the current scan plane
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

                # If the ray hit the surface, calculate the 4x4 frame
                if len(locations) > 0:
                    point = locations[0]
                    tri_index = index_tri[0]
                    z_axis = mesh.face_normals[tri_index].copy()

                    # Normalize the Z-axis (Tool Vector)
                    z_axis /= np.linalg.norm(z_axis)

                    # Define a feed/travel direction vector (X-axis)
                    x_axis = np.array([1.0, 0.0, 0.0])
                    if np.abs(np.dot(z_axis, x_axis)) > 0.95:
                        x_axis = np.array([0.0, 1.0, 0.0])

                    # Use cross products to ensure all axes are perfectly 90 degrees apart
                    y_axis = np.cross(z_axis, x_axis)
                    y_axis /= np.linalg.norm(y_axis)

                    x_axis = np.cross(y_axis, z_axis)
                    x_axis /= np.linalg.norm(x_axis)

                    # 5. Build the 4x4 Homogeneous Transformation Matrix
                    frame = np.identity(4)
                    frame[0:3, 0] = x_axis
                    frame[0:3, 1] = y_axis
                    frame[0:3, 2] = z_axis
                    frame[0:3, 3] = point

                    trajectory_frames.append(frame)

    return trajectory_frames

def visualize_with_trimesh(mesh_path):
    # 1. Load the base CAD part
    mesh = trimesh.load(mesh_path, force='mesh')
    
    # 2. Initialize your empty list for the trajectory frames
    trajectory_frames = generate_trajectory_frames(mesh, 500)

    # 3. Collect all geometries to visualize
    geometries = [mesh]
    for frame in trajectory_frames:
        # Create an RGB coordinate axis visualization marker
        axis_visual = trimesh.creation.axis(
            origin_size=1.0, axis_radius=0.5, axis_length=12.0, transform=frame
        )
        geometries.append(axis_visual)

    # 4. Launch the interactive display window with the mesh and tool axes
    trimesh.Scene(geometries).show()

def visualize_points_only(mesh_path):
    # 1. Load the mesh and generate trajectory
    mesh = trimesh.load(mesh_path, force='mesh')
    trajectory_frames = generate_trajectory_frames(mesh, 50)
    
    # 2. Extract origin points (XYZ) from the 4x4 transformation matrices
    points = np.array([frame[:3, 3] for frame in trajectory_frames])
    
    # 3. Create a PointCloud geometry and show it without the mesh
    point_cloud = trimesh.points.PointCloud(points)
    trimesh.Scene(point_cloud).show()

def visualize_spheres_at_points(mesh_path, sphere_radius=1.0):
    """
    Visualizes the trajectory points as actual 3D spheres instead of square pixels.
    """
    # 1. Load the mesh and generate trajectory
    mesh = trimesh.load(mesh_path, force='mesh')
    trajectory_frames = generate_trajectory_frames(mesh, 500)
    
    # 2. Extract origin points
    points = np.array([frame[:3, 3] for frame in trajectory_frames])
    
    # 3. Create a list of small spheres at each point
    geometries = []
    for p in points:
        sphere = trimesh.creation.uv_sphere(radius=sphere_radius)
        sphere.apply_translation(p)
        sphere.visual.face_colors = [0, 0, 0, 255]  # Make them black
        geometries.append(sphere)
    
    # 4. Show the collection of spheres
    trimesh.Scene(geometries).show()

if __name__ == "__main__":
    # Example usage
    mesh_path = "C:\\Users\\shish\\C-\\Slicing\\StepRobotFrames\\Adapter.stl"
    #visualize_with_trimesh(mesh_path)
    visualize_spheres_at_points(mesh_path, sphere_radius=0.5)