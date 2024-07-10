import numpy as np
import pysocialforce as psf


class Human_SF:
    def __init__(self, obs=None):
        self.obs = obs
        self.reset()

    def update(self):
        # Use the simulator to update the human's position
        self.sim.step()

        pedestrian_states, group_states = (
            self.sim.get_states()
        )  # Get the state of the pedestrians and groups
        latest_state = pedestrian_states[-1][0]

        # Update the position and orientation
        self.pos = np.array([latest_state[0], latest_state[1]], dtype=np.float32)
        self.theta = np.arctan2(latest_state[3], latest_state[2])
        self.vx = latest_state[2]
        self.vy = latest_state[3]

    def reset(self):
        # Set the initial position of the human within the range [-10.0, 10.0]
        # x = np.float32(np.random.uniform(-10.0, 10.0))
        # y = np.float32(np.random.uniform(-10.0, 10.0))
        # vx = np.float32(0.0)  # Initial velocity x-component
        # vy = np.float32(0.0)  # Initial velocity y-component
        # dx = np.float32(np.random.uniform(0.1, 10.0)) # goal_x
        # dy = np.float32(np.random.uniform(0.1, 10.0)) # goal_y

        # initial_state = np.array([[x, y, vx, vy, dx, dy]])
        # print("initial state: ", initial_state)

        self.initial_state = np.array(
            [
                # [0.0, 10, 0.5, 0.5, 0.2, 0.1],
                # [0.5, 10, -0.5, -0.5, 0.5, 0.0],
                # [0.0, 0.0, 0.0, 0.5, 1.0, 10.0],
                # [1.0, 0.0, 0.0, 0.5, 2.0, 10.0],
                [-10.0, -10.0, 0.0, 0.5, 18.0, 18.0],
                # [3.0, 0.0, 0.0, 0.5, 4.0, 10.0],
            ]
        )

        groups = None

        self.sim = psf.Simulator(
            self.initial_state,
            groups=groups,
            obstacles=self.obs,
            config_file="/home/sriram/gym_play/PySocialForce/examples/example.toml",
        )

        self.pos = np.array(
            [self.initial_state[0][0], self.initial_state[0][1]], dtype=np.float32
        )
        self.theta = np.arctan2(self.initial_state[0][3], self.initial_state[0][2])
        self.vx = self.initial_state[0][2]
        self.vy = self.initial_state[0][3]

    def get_obstacles(self):
        return self.sim.get_obstacles()

    def get_goal_pos(self):
        return np.array(
            [self.initial_state[0][4], self.initial_state[0][5]], dtype=np.float32
        )

    def get_velocities(self):
        return self.vx, self.vy
