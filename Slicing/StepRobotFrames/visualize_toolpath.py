import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
from cad_processor import CADProcessor

def visualize_frames(processor: CADProcessor, axis_length: float = 2.0):
    """
    Visualizes the generated robotic frames in 3D.
    Red = X-axis, Green = Y-axis, Blue = Z-axis (Tool axis)
    """
    if not processor.frames:
        print("No frames to visualize.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract origins for the main path line
    origins = np.array([[f.origin.x, f.origin.y, f.origin.z] for f in processor.frames])
    
    # Plot the sequential path (dotted line)
    ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], 'k--', alpha=0.3, label='Toolpath')

    for f in processor.frames:
        o = np.array([f.origin.x, f.origin.y, f.origin.z])
        
        # Define axis end points
        x_tail = o + np.array([f.x_axis.x, f.x_axis.y, f.x_axis.z]) * axis_length
        y_tail = o + np.array([f.y_axis.x, f.y_axis.y, f.y_axis.z]) * axis_length
        z_tail = o + np.array([f.z_axis.x, f.z_axis.y, f.z_axis.z]) * axis_length

        # Draw local X (Red)
        ax.plot([o[0], x_tail[0]], [o[1], x_tail[1]], [o[2], x_tail[2]], color='r', linewidth=1.5)
        # Draw local Y (Green)
        ax.plot([o[1], y_tail[0]], [o[1], y_tail[1]], [o[2], y_tail[2]], color='g', linewidth=1.5)
        # Draw local Z (Blue - Tool Axis)
        ax.plot([o[0], z_tail[0]], [o[1], z_tail[1]], [o[2], z_tail[2]], color='b', linewidth=2.0)

    # Labeling
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'6-Axis Toolpath Visualization ({len(processor.frames)} frames)\nRed=X, Green=Y, Blue=Z(Tool)')
    
    # Equalize aspect ratio (crucial for seeing correct orientations)
    max_range = np.array([origins[:,0].max()-origins[:,0].min(), 
                          origins[:,1].max()-origins[:,1].min(), 
                          origins[:,2].max()-origins[:,2].min()]).max() / 2.0

    mid_x = (origins[:,0].max()+origins[:,0].min()) * 0.5
    mid_y = (origins[:,1].max()+origins[:,1].min()) * 0.5
    mid_z = (origins[:,2].max()+origins[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 1. Initialize Processor
    proc = CADProcessor()
    
    # 2. Configure Path (Update to your actual file)
    step_file = Path(r"C:\Users\shish\OneDrive\Desktop\CAD\LeadScrew Nut 8mm x 2mmPitch.STEP")
    
    if step_file.exists():
        # 3. Process CAD with a standoff to see the frames clearly above the surface
        print(f"Processing {step_file.name}...")
        proc.load_and_process(step_file, samples_u=6, samples_v=6, stand_off=10.0)
        
        # 4. Optional: Transform to a specific workspace area
        proc.apply_global_transform(rotation_deg=[0, 0, 0], translation=[0, 0, 0])
        
        # 5. Visualize
        visualize_frames(proc, axis_length=5.0)
    else:
        print(f"File not found: {step_file}")