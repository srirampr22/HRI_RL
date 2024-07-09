import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from Human import Human
from Robot import Robot

class HumanRobotEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self):
        super(HumanRobotEnv, self).__init__()
        
        self.env_size = 80.0
        # self.desired_distance = np.float32(np.random.uniform(3.0, 7.0)) # This can be randomly set
        self.desired_distance = 5.0 # Object lenght (the distance the robot needs to maintain from the human)
        
        self.robot_pos_history = []
        self.human_pos_history = []
        
        # defining action space (continuous or discrete) the actions that the agent can take is in the form of 2d twist (linear and angular velocity)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # defining observation space
        # [robot_x, robot_y, robot_theta, human_x, human_y, human_theta]     
        obs_low = np.array([-self.env_size, -self.env_size, -np.pi, -self.env_size, -self.env_size, -np.pi], dtype=np.float32)
        obs_high = np.array([self.env_size, self.env_size, np.pi, self.env_size, self.env_size, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        self.human = Human()
        
        self.robot = Robot(obj_lenght=self.desired_distance, human_pos=self.human.pos)
        
        # Initialize the state of the environment
        self.reset()
        
        
    def step(self, action):
        
        terminated = False  
        truncated = False 
        
        # define step function
        linear_vel, angular_vel = action
        self.robot.theta += angular_vel
        self.robot.pos += linear_vel * np.array([np.cos(self.robot.theta), np.sin(self.robot.theta)])
        
        # Update human position and orientation
        self.human.update()
        
        self.robot_pos_history.append(self.robot.pos.copy())
        self.human_pos_history.append(self.human.pos.copy())
        
        # distance to the human
        distance_to_human = np.linalg.norm(self.robot.pos - self.human.pos)
        # orientation difference to the human
        orientation_difference = np.abs(self.robot.theta - self.human.theta)

        # Reward components
        translation_reward = -np.exp(-0.5 * ((distance_to_human - self.desired_distance) ** 2))
        rotation_reward = -orientation_difference  
        
        # Total reward
        reward = translation_reward + rotation_reward
        
        
        # Termination condition: if robot moves out of bounds
        if np.any(self.human.pos < -self.env_size) or np.any(self.human.pos > self.env_size):
            terminated = True

        # Info can be used for debugging
        info = {
            'distance_to_human': distance_to_human,
            'orientation_difference': orientation_difference
        }
        
        state = np.concatenate([self.robot.pos, [self.robot.theta], self.human.pos, [self.human.theta]]).astype(np.float32)
        
        return state, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        # define reset function
        if seed is not None:
            np.random.seed(seed)
            

        self.desired_distance = 5.0
        
        self.robot_pos_history = []
        self.human_pos_history = []
        
    
        self.human.reset()
        self.robot.reset(obj_lenght=self.desired_distance, human_pos=self.human.pos)
        
        # self.robot_pos_history.append(self.robot.pos.copy())
        # self.human_pos_history.append(self.human.pos.copy())

        
        state = np.concatenate([self.robot.pos, [self.robot.theta], self.human.pos, [self.human.theta]]).astype(np.float32)
        
        info = {}
        return state, info
    
    def render(self, mode='human'):
        plt.clf()
        
        arrow_size = 3.0
        
        robot_positions = np.array(self.robot_pos_history)
        human_positions = np.array(self.human_pos_history)
        if len(robot_positions) > 1:
            plt.plot(robot_positions[:, 0], robot_positions[:, 1], 'r-', label='Robot Trajectory')
        if len(human_positions) > 1:
            plt.plot(human_positions[:, 0], human_positions[:, 1], 'b-', label='Human Trajectory')
            
        # Plot the robot with an arrow indicating orientation
        plt.arrow(self.robot.pos[0], self.robot.pos[1], 0.5 * np.cos(self.robot.theta), 0.5 * np.sin(self.robot.theta),
                  head_width=arrow_size, head_length=arrow_size, fc='r', ec='r')
        # Plot the human with an arrow indicating orientation
        plt.arrow(self.human.pos[0], self.human.pos[1], 0.5 * np.cos(self.human.theta), 0.5 * np.sin(self.human.theta),
                  head_width=arrow_size, head_length=arrow_size, fc='b', ec='b')
        # Plot the distance between the robot and the human as a line
        plt.plot([self.robot.pos[0], self.human.pos[0]], [self.robot.pos[1], self.human.pos[1]], 'g--')
        plt.xlim(-self.env_size+10, self.env_size+10)
        plt.ylim(-self.env_size+10, self.env_size+10)
        plt.legend()
        plt.pause(0.01)
        
    def close(self):
        plt.close()  
        
