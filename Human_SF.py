import numpy as np
import pysocialforce as psf


class Human_SF:
    def __init__(self):
        self.reset()
        

    def update(self):
        # Use the simulator to update the human's position
        self.sim.step()
        # state = self.sim.get_states()  # Get the state of the first (and only) pedestrian
        pedestrian_states, group_states = self.sim.get_states()  # Get the state of the pedestrians and groups
        latest_state = pedestrian_states[-1][0]
        # print("sim state: ", latest_state)
        
        # Update the position and orientation
        self.pos = np.array([latest_state[0], latest_state[1]], dtype=np.float32)
        self.theta = np.arctan2(latest_state[3], latest_state[2])
        # print(self.sim.get_states())
        


    def reset(self):
        # Set the initial position of the human within the range [-10.0, 10.0]
        # x = np.float32(np.random.uniform(-10.0, 10.0))
        # y = np.float32(np.random.uniform(-10.0, 10.0))
        # vx = np.float32(0.0)  # Initial velocity x-component
        # vy = np.float32(0.0)  # Initial velocity y-component
        # dx = np.float32(np.random.uniform(0.1, 10.0))
        # dy = np.float32(np.random.uniform(0.1, 10.0))
        
        # initial_state = np.array([[x, y, vx, vy, dx, dy]])
        # print("initial state: ", initial_state)
        
        
        initial_state = np.array(
        [
            # [0.0, 10, 0.5, 0.5, 0.2, 0.1],
            # [0.5, 10, -0.5, -0.5, 0.5, 0.0],
            # [0.0, 0.0, 0.0, 0.5, 1.0, 10.0],
            # [1.0, 0.0, 0.0, 0.5, 2.0, 10.0],
            [2.0, 0.0, 0.0, 0.5, 3.0, 10.0],
            # [3.0, 0.0, 0.0, 0.5, 4.0, 10.0],
        ]
        )
         
        groups = None
        # obs = None
        obs = [[-1, -1, -1, 11], [3, 3, -1, 11]]

        self.sim = psf.Simulator(
        initial_state,
        groups=groups,
        obstacles=obs,
        config_file="/home/sriram/gym_play/PySocialForce/examples/example.toml",
        )
        
        self.pos = np.array([initial_state[0][0], initial_state[0][1]], dtype=np.float32)
        self.theta = np.arctan2(initial_state[0][3], initial_state[0][2])
        

        # Create the initial state for the simulator
        # initial_state = np.array([[x, y, vx, vy, dx, dy]])
        # self.sim = Simulator(initial_state)
        
    def get_obstacles(self):
        return self.sim.get_obstacles()