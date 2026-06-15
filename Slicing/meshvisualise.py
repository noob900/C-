import numpy as np
from scipy.interpolate import make_interp_spline, splprep, splev
from scipy.interpolate import splprep, splev
import trimesh

def generate_trajectory_frames(mesh, num_points):

    trajectory_frames = []
  
    bounding_box = mesh.bounds
    # We divide by 3 because we are performing 3 orthogonal scans
    res = max(2, int(np.sqrt(num_points / 3)))
    
    # Create grids for all three axes
    x_grid = np.linspace(bounding_box[0][0], bounding_box[1][0], res)
    y_grid = np.linspace(bounding_box[0][1], bounding_box[1][1], res)
    z_grid = np.linspace(bounding_box[0][2], bounding_box[1][2], res)

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


def generate_bezier_path(mesh_path, num_points=500):
    mesh = trimesh.load(mesh_path, force='mesh')
    trajectory_frames = generate_trajectory_frames(mesh, 500)
    points = np.array([frame[:3, 3] for frame in trajectory_frames])

    if len(points) < 4:
        raise ValueError("Please provide at least 4 points to create a smooth cubic curve.")

    t_input = np.linspace(0, 1, len(points))

    spline_engine = make_interp_spline(t_input, points, k=3) # k=3 enforces a cubic polynomial

    t_dense = np.linspace(0, 1, num_points)

    smooth_path = spline_engine(t_dense)
    bezier_line = trimesh.load_path(smooth_path)
    bezier_line.colors = [[0, 255, 0, 255]]  # Bright Green [R, G, B, Alpha]

    scene = trimesh.Scene(bezier_line)
    scene.show()



def generate_smooth_layered_path(mesh_path, num_layers=30, points_per_layer=120, smoothing=0.1):
    """
    Generates a high-fidelity layered toolpath without showing the original mesh.
    Uses periodic splines for perfectly smooth loop closures.
    """
    mesh = trimesh.load(mesh_path, force='mesh')
    bounds = mesh.bounds
    centroid = mesh.centroid
    
    # 1. Higher resolution Z-levels
    z_levels = np.linspace(bounds[0][2], bounds[1][2], num_layers)
    
    # Start with an empty list (omitting the mesh) as requested
    geometries = []
    
    for z in z_levels:
        layer_hits = []
        # 2. Increased radial resolution for better detail capture
        # We use endpoint=False because the periodic spline handles the wrap-around
        angles = np.linspace(0, 2 * np.pi, points_per_layer, endpoint=False)
        radius = np.linalg.norm(bounds[1] - bounds[0]) * 1.1
        
        for angle in angles:
            origin = [
                centroid[0] + radius * np.cos(angle),
                centroid[1] + radius * np.sin(angle),
                z
            ]
            direction = [centroid[0] - origin[0], centroid[1] - origin[1], 0]
            direction /= np.linalg.norm(direction)
            
            locations, _, _ = mesh.ray.intersects_location(
                ray_origins=np.array([origin]),
                ray_directions=np.array([direction])
            )
            
            if len(locations) > 0:
                layer_hits.append(locations[0])
        
        # 3. Apply Parametric Spline (splprep) for superior smoothness
        if len(layer_hits) > 10:
            layer_hits = np.array(layer_hits).T  # splprep expects (N_dim, N_points)
            
            # splprep finds a smooth parametric representation (x(u), y(u), z(u))
            # s: smoothing factor (higher = smoother/less accurate to raw hits)
            # per=True: ensures C2 continuity at the loop closure
            tck, u = splprep(layer_hits, s=smoothing, per=True)
            
            # Evaluate the spline at high density
            u_fine = np.linspace(0, 1, 400)
            smooth_points = np.array(splev(u_fine, tck)).T
            
            # Close the loop visually
            smooth_points = np.vstack([smooth_points, smooth_points[0]])
            
            path = trimesh.load_path(smooth_points)
            path.colors = [[0, 255, 0, 255]] 
            geometries.append(path)
            
    trimesh.Scene(geometries).show()

if __name__ == "__main__":
    # Example usage
    mesh_path = "C:\\Users\\shish\\C-\\Slicing\\StepRobotFrames\\Adapter.stl"
    #visualize_with_trimesh(mesh_path)
    generate_smooth_layered_path(mesh_path)
    #visualize_spheres_at_points(mesh_path, sphere_radius=0.5)