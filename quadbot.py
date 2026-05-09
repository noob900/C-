import pybullet as p
import pybullet_data
import time

# 1. Setup the Simulation Environment
physicsClient = p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Path for floor/basic models
p.setGravity(0, 0, -9.8)
planeId = p.loadURDF("plane.urdf")

# 2. Load your Custom Quadbot URDF
# Using custom URDF file in the same directory
quadbotId = p.loadURDF("quadbot_custom.urdf", [0,0,0.3])

# 3. Create Sliders for the Joints
# We get the number of joints and create a slider for each 'revolute' joint
num_joints = p.getNumJoints(quadbotId)
slider_ids = []

for i in range(num_joints):
    joint_info = p.getJointInfo(quadbotId, i)
    joint_name = joint_info[1].decode("utf-8")
    # Only add sliders for joints that can actually rotate (Type 0)
    if joint_info[2] == 0: 
        slider_id = p.addUserDebugParameter(joint_name, -1.5, 1.5, 0)
        slider_ids.append((i, slider_id))

# 4. Main Loop: Link Sliders to Robot Movement
while True:
    for joint_index, slider_id in slider_ids:
        target_pos = p.readUserDebugParameter(slider_id)
        p.setJointMotorControl2(quadbotId, joint_index, p.POSITION_CONTROL, target_pos)
    
    p.stepSimulation()
    time.sleep(1./240.)