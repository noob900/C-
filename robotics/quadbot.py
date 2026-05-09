import pybullet as p
import pybullet_data
import time
import os
import math

# 1. Setup the Simulation Environment
physicsClient = p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Path for floor/basic models
p.setGravity(0, 0, -9.8)
planeId = p.loadURDF("plane.urdf")

URDF_PATH = r"C:\Users\shish\C-\robotics\quadbot_custom.urdf"
last_modified_time = os.path.getmtime(URDF_PATH)

def load_robot():
    """Loads or reloads the robot and recreates sliders."""
    p.removeAllUserParameters()
    # Add global gait controls
    auto_id = p.addUserDebugParameter("Auto Mode (0/1)", 0, 1, 0)
    freq_id = p.addUserDebugParameter("Gait Frequency", 0, 10, 2)
    amp_id = p.addUserDebugParameter("Gait Amplitude", 0, 1, 0.4)
    
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0.3])
    num_joints = p.getNumJoints(robot_id)
    sliders = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode("utf-8")
        if joint_info[2] == 0:  # Revolute joint
            slider_id = p.addUserDebugParameter(joint_name, -1.5, 1.5, 0)
            sliders.append((i, slider_id, joint_name))
    return robot_id, sliders, (auto_id, freq_id, amp_id)

quadbotId, slider_ids, gait_params = load_robot()

# 4. Main Loop: Link Sliders to Robot Movement
while True:
    # Check if the URDF file has been modified
    current_mtime = os.path.getmtime(URDF_PATH)
    if current_mtime > last_modified_time:
        print("URDF change detected! Reloading...")
        p.removeBody(quadbotId)
        quadbotId, slider_ids, gait_params = load_robot()
        last_modified_time = current_mtime

    # Read gait control parameters
    is_auto = p.readUserDebugParameter(gait_params[0]) > 0.5
    freq = p.readUserDebugParameter(gait_params[1])
    amp = p.readUserDebugParameter(gait_params[2])
    t = time.time()

    for joint_index, slider_id, joint_name in slider_ids:
        try:
            if is_auto:
                # Basic procedural gait: diagonal coordination
                # Front-Left and Rear-Right are in sync; others are 180 degrees out of phase
                phase_offset = 0
                if "front_right" in joint_name or "rear_left" in joint_name:
                    phase_offset = math.pi
                
                target_pos = amp * math.sin(t * freq + phase_offset)
            else:
                target_pos = p.readUserDebugParameter(slider_id)

            p.setJointMotorControl2(quadbotId, joint_index, p.POSITION_CONTROL, target_pos)
        except:
            # Handle cases where sliders are being recreated during a reload
            continue
    
    p.stepSimulation()
    time.sleep(1./240.)