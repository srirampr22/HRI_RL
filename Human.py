import numpy as np


class Human:
    def __init__(self):
        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        self.theta = np.float32(np.random.uniform(-np.pi, np.pi))
        self.reset()

    def update(self):
        self.time += self.delta_time
        x = self.time
        y = self.amplitude * np.sin(self.frequency * x + self.phase)
        self.pos = np.array([x, y], dtype=np.float32)
        self.theta = np.arctan2(y, 1.0)  # Simple approximation of orientation

    def reset(self):
        # self.pos = np.array([0.0, 0.0], dtype=np.float32)
        # self.theta = np.float32(np.random.uniform(-np.pi, np.pi))
        self.amplitude = np.float32(np.random.uniform(1.0, 2.0)) 
        self.frequency = np.float32(np.random.uniform(0.5, 1.0)) 
        self.phase = np.float32(np.random.uniform(0.0, 2 * np.pi)) 
        self.delta_time = np.random.uniform(0.5, 1.0)
        
        # self.pos = self.init_pos
        # self.theta = self.init_theta
        self.time = 0.0