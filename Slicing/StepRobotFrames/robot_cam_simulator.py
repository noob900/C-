import pybullet as p
import pybullet_data
import time
import numpy as np
from pathlib import Path
from step_to_robot_frames import generate_frames

def run_simulation():
    # 1. Setup PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # 2. Load 6-Axis Robot (Using a KUKA as an example from pybullet_data)
    # In your real app, you would load your custom 6-axis/7-axis URDF here.
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    
    # Identify end-effector link (usually the last link)
    ee_link_index = num_joints - 1

    # 3. Generate Toolpath from STEP
    step_file = Path(r"C:\Users\shish\OneDrive\Desktop\CAD\Cube.step")
    if not step_file.exists():
        print(f"CAD File not found: {step_file}")
        return

    print("Processing STEP file for toolpath...")
    frames = generate_frames(
        step_path=step_file,
        samples_u=5,
        samples_v=5,
        trim_margin=0.05,
        tolerance=1e-6,
        tool_axis="toward_surface",
        ref_axis_name="world_z",
        stand_off=0.02  # 20mm standoff
    )

    # 4. Simulation Loop
    print(f"Simulating {len(frames)} frames...")
    
    # Visual marker for the toolpath
    for f in frames:
        p.addUserDebugLine(
            [f.surface_point.x, f.surface_point.y, f.surface_point.z],
            [f.origin.x, f.origin.y, f.origin.z],
            [1, 0, 0], lifespan=30
        )

    for frame in frames:
        # Target position and orientation (Quaternion)
        target_pos = [frame.origin.x, frame.origin.y, frame.origin.z]
        # Convert Euler angles to Quaternion for PyBullet
        target_orn = p.getQuaternionFromEuler([
            np.radians(frame.roll_deg), 
            np.radians(frame.pitch_deg), 
            np.radians(frame.yaw_deg)
        ])

        # 5. Inverse Kinematics (The "RoboDK" core)
        joint_poses = p.calculateInverseKinematics(
            robot_id, 
            ee_link_index, 
            target_pos, 
            target_orn,
            residualThreshold=1e-4
        )

        # Apply joint positions to the robot
        for i in range(len(joint_poses)):
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i]
            )

        p.stepSimulation()
        time.sleep(0.05)

    print("Simulation Complete.")
    while True: p.stepSimulation()

if __name__ == "__main__":
    run_simulation()