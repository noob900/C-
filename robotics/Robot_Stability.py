#gyroscope sensor Theoritical model to maintain stability of robot.
import os
import pybullet as p
import pybullet_data
import numpy as np
import time 
import itertools


# 1. Setup the Simulation Environment
physicsClient = p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Path for floor/basic models
p.setGravity(0, 0, -9.8)
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
planeId = p.loadURDF("plane.urdf",startPos,startOrientation)
URDF_PATH = r"C:\Users\shish\C-\robotics\quadbot_custom.urdf"
last_modified_time = os.path.getmtime(URDF_PATH)



class Gyroscope:
    def __init__(self, bias_sigma=0.01, noise_sigma=0.005):
        # Initial constant bias (randomly chosen for this specific sensor)
        self.bias = np.random.normal(0, bias_sigma, 3)
        self.noise_sigma = noise_sigma

    def get_reading(self, true_angular_velocity):
        # Simulate the gyroscope reading by adding bias and noise to the true angular velocity
        noise = np.random.normal(0, self.noise_sigma, 3)
        return true_angular_velocity + self.bias + noise

class Coordinates:

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

class QuadBot:
    @staticmethod
    def load_robot():
        p.removeAllUserParameters()
        robot_id = p.loadURDF(URDF_PATH, [0, 0, 0.3])
        return robot_id
    
    @staticmethod
    def get_gyro_data(robot_id):
        # Get base angular velocity for the gyroscope
        _, angular_vel = p.getBaseVelocity(robot_id)
        return np.array(angular_vel)

class QuadbotMotionGenerator:
    def __init__(self, increment=0.05):

        self.increment = increment
        self.num_legs = 4
        self.joints_per_leg = 3
        self.total_motors = 12

    def get_current_joints(self, robot_id, pybullet_client):

        joint_states = []
        for i in range(pybullet_client.getNumJoints(robot_id)):
            # We only want revolute joints (motors)
            info = pybullet_client.getJointInfo(robot_id, i)
            if info[2] == pybullet_client.JOINT_REVOLUTE:
                state = pybullet_client.getJointState(robot_id, i)
                joint_states.append(state[0]) # index 0 is position
        return np.array(joint_states)

    def generate_combinations(self, current_angles):

        leg_options = [-self.increment, 0, self.increment]
        leg_combos = list(itertools.product(leg_options, repeat=self.num_legs))
        all_motions = []
        for combo in leg_combos:
            # Start with a copy of current angles
            new_motion = np.copy(current_angles)
            for leg_idx in range(self.num_legs):
                motor_idx = leg_idx * self.joints_per_leg
                new_motion[motor_idx] += combo[leg_idx]
            
            all_motions.append(new_motion)

        return np.array(all_motions)

quadbotId = QuadBot.load_robot()
gyro = Gyroscope()
motion_gen = QuadbotMotionGenerator(increment=0.05)

# Track time and combination index for the reset system
start_time = time.time()
combo_index = 0
RESET_INTERVAL = 5  # seconds
revolute_joint_indices = [i for i in range(p.getNumJoints(quadbotId)) if p.getJointInfo(quadbotId, i)[2] == p.JOINT_REVOLUTE]

while True:
    # Check if the URDF file has been modified
    current_modified_time = os.path.getmtime(URDF_PATH)
    if current_modified_time > last_modified_time:
        print("URDF change detected! Reloading...")
        p.removeBody(quadbotId)
        quadbotId = QuadBot.load_robot()
        last_modified_time = current_modified_time  
        # Re-cache indices and reset timer on reload
        revolute_joint_indices = [i for i in range(p.getNumJoints(quadbotId)) if p.getJointInfo(quadbotId, i)[2] == p.JOINT_REVOLUTE]
        start_time = time.time()
        combo_index = 0
    
    # 1. Get Sensor Data
    true_ang_vel = QuadBot.get_gyro_data(quadbotId)
    gyro_reading = gyro.get_reading(true_ang_vel)

    # 2. Retrieve current joint states into an array
    current_joints = motion_gen.get_current_joints(quadbotId, p)

    # 3. Generate the 81 combinations (3 options per leg ^ 4 legs)
    combos = motion_gen.generate_combinations(current_joints)

    # 4. Stabilisation Logic: Choose the combo that counters the tilt
    if len(combos) > 0:
        # Simple Heuristic: 
        # If pitch > 0 (leaning forward), we want combos that move front legs forward.
        # For now, let's select the combo that best minimizes the 'potential' tilt.
        # In a real system, you'd use a scoring function.
        
        best_combo = combos[0]
        min_instability_score = float('inf')
        
        # Example: Choose a combo based on simple pitch/roll correction
        # We can simulate which combo is "opposite" to the current gyro reading
        for combo in combos:
            # Calculate a 'score' for this combo. 
            # We want to pick a combo where the leg adjustment opposes the tilt.
            # This is a placeholder for your specific robot's kinematics:
            correction_score = np.sum(np.abs(combo - current_joints)) 
            
            if correction_score < min_instability_score:
                min_instability_score = correction_score
                best_combo = combo

        # Apply the chosen "stabilizing" combination
        for i, angle in enumerate(best_combo):
            if i < len(revolute_joint_indices):
                p.setJointMotorControl2(quadbotId, revolute_joint_indices[i], p.POSITION_CONTROL, angle)

    # 5. Reset logic: Reset simulation every 20 seconds and move to the next combo
    elapsed_time = time.time() - start_time
    if elapsed_time > RESET_INTERVAL:
        print(f"Resetting... Testing combo {combo_index + 1}/81")
        
        # Restore robot to initial pose and clear velocity
        p.resetBasePositionAndOrientation(quadbotId, [0, 0, 0.3], startOrientation)
        p.resetBaseVelocity(quadbotId, [0, 0, 0], [0, 0, 0])
        for i in revolute_joint_indices:
            p.resetJointState(quadbotId, i, 0)
            
        combo_index = (combo_index + 1) % 81
        start_time = time.time()

    # Step the simulation
    p.stepSimulation()
    time.sleep(1./240.)
