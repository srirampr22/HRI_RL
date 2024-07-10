import numpy as np

class RobotDiff:
    def __init__(self, config_file, human_pos):
        if config_file is None:
            raise ValueError("Robot Configuration file not found")

        self.wheel_radius = config_file["wheel_radius"]# Radius of the wheels
        self.wheel_base = config_file["wheel_base"]     # Distance between the wheels
        self.desired_dist = config_file["desired_dist"]     
        self.robot_radius = config_file["robot_radius"]  # Radius of the robot
        self.reset(human_pos)
        
    def update(self, action):
        # Action is now [v_left, v_right] - velocities of left and right wheels
        v_left, v_right = action
        
        # Calculate linear and angular velocities
        linear_vel = self.wheel_radius * (v_right + v_left) / 2
        angular_vel = self.wheel_radius * (v_right - v_left) / self.wheel_base
        
        # Update orientation
        self.theta += angular_vel
        
        # Update position
        self.pos += linear_vel * np.array([np.cos(self.theta), np.sin(self.theta)])
        
        # Calculate velocity components
        self.vx = linear_vel * np.cos(self.theta)
        self.vy = linear_vel * np.sin(self.theta)
        
        # Store linear and angular velocities for potential use elsewhere
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel
        
    def reset(self, human_pos):
        self.theta = np.float32(np.random.uniform(-np.pi, np.pi))
        dist_away = self.desired_dist + self.robot_radius
        self.init_pos = human_pos - dist_away * np.array([np.cos(self.theta), np.sin(self.theta)])
        self.pos = self.init_pos.copy()
        self.vx = 0.0
        self.vy = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0

    def get_velocities(self):
        return self.linear_vel, self.angular_vel