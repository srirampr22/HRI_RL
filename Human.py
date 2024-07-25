import numpy as np
from pysocialforce import Simulator


class Human:
    def __init__(self):
        # self.pos = np.array([0.0, 0.0], dtype=np.float32)
        # self.theta = np.float32(np.random.uniform(-np.pi, np.pi))
        self.reset()

    def update(self):
        self.time += self.delta_time
        delta_x = self.time
        delta_y = self.amplitude * np.sin(self.frequency * delta_x + self.phase)
        self.pos = self.initial_pos + np.array([delta_x, delta_y], dtype=np.float32)
        self.theta = np.arctan2(delta_y, delta_x)   # Simple approximation of orientation
        
        # self.theta = np.float32(np.random.uniform(-np.pi, np.pi))

    def reset(self):
        # self.pos = np.array([0.0, 0.0], dtype=np.float32)
        x = np.float32(np.random.uniform(0.0, -60.0))
        y = np.float32(np.random.uniform(-40.0, -60.0))
        # x = 30.0
        # y = 30.0
        self.initial_pos = np.array([x, y], dtype=np.float32)  # Save the initial position
        self.pos = self.initial_pos.copy()  
        # print(f"Human reset to position: {self.pos}")

        
        self.theta = np.float32(np.random.uniform(-np.pi, np.pi))
        self.amplitude = np.float32(np.random.uniform(1.0, 2.0)) 
        self.frequency = np.float32(np.random.uniform(0.5, 1.0)) 
        self.phase = np.float32(np.random.uniform(0.0, 2 * np.pi)) 
        self.delta_time = np.random.uniform(0.5, 1.0)
        
        # self.pos = self.init_pos
        # self.theta = self.init_theta
        self.time = 0.0