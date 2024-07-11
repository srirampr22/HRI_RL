import numpy as np

class RobotDiff:
    def __init__(self, config_file, human_state):
        if config_file is None:
            raise ValueError("Robot Configuration file not found")

        self.wheel_radius = config_file["wheel_radius"]# Radius of the wheels
        self.wheel_base = config_file["wheel_base"]     # Distance between the wheels
        self.desired_dist = config_file["desired_dist"]     
        self.robot_radius = config_file["robot_radius"]  # Radius of the robot
        self.step_width = config_file["step_width"]      # Step width
        self.reset(human_state)
        
    def update(self, action):
        """A differential drive robot has two wheels, each of which can be controlled independently. 
        The robot's movement is determined by the velocities of these wheels."""
        # Action is now [v_left, v_right] - velocities of left and right wheels
        v_left, v_right = action
        
        # Linear velocity is the average of the left and right wheel velocities
        linear_vel = self.wheel_radius * (v_right + v_left) / 2
        # Angular velocity is the difference between the right and left wheel velocities divided by the wheel base
        angular_vel = self.wheel_radius * (v_right - v_left) / self.wheel_base

        # The robot's orientation (theta) is updated based on its angular velocity and the time step (step_width)
        self.theta += angular_vel * self.step_width
        self.theta = self.wrap_to_pi(self.theta)

        # The robot's position (pos) is updated based on its linear velocity, orientation (theta), and time step (step_width)
        self.pos += linear_vel * self.step_width * np.array([np.cos(self.theta), np.sin(self.theta)])

        # The velocity components in the x and y directions are computed using the linear velocity and the orientation (theta)
        self.vx = linear_vel * np.cos(self.theta)
        self.vy = linear_vel * np.sin(self.theta)
        
        # Store linear and angular velocities for potential use elsewhere
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel
        
    def reset(self, human_state):
        # self.theta = np.float32(np.random.uniform(-np.pi, np.pi))
        human_pos = human_state[0:2]
        human_theta = human_state[2]
        human_vx, human_vy = human_state[3:5]
        human_goal_pos = human_state[6:8]
        
        self.theta = human_theta
        dist_away = self.desired_dist + self.robot_radius
        self.init_pos = human_pos - dist_away * np.array([np.cos(self.theta), np.sin(self.theta)])
        self.goal_pos = human_goal_pos - self.desired_dist * np.array([np.cos(self.theta), np.sin(self.theta)])
        self.pos = self.init_pos.copy()
        self.vx = 0.0
        self.vy = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0

    def get_velocities(self):
        return self.linear_vel, self.angular_vel
    
    def get_goal_pos(self):
        return self.goal_pos
    
    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi