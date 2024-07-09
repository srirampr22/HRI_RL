import numpy as np

class Robot:
    def __init__(self, obj_lenght, human_pos):
        self.obj_lenght =  obj_lenght
        # self.pos = np.array([0.0, 0.0], dtype=np.float32)
        # self.theta = 0.0
        self.reset(obj_lenght, human_pos)
        
    

    def reset(self, obj_lenght, human_pos):
        self.theta =  np.float32(np.random.uniform(-np.pi, np.pi))
        self.pos = human_pos - self.obj_lenght * np.array([np.cos(self.theta), np.sin(self.theta)])