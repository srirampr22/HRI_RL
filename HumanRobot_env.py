import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from Human import Human
from Human_SF import Human_SF
from Robot import Robot
from RobotDiff import RobotDiff
from Rewards import RewardCalculator
from scipy.spatial import KDTree
import math
from math import pi

def get_obstacle_points(robot_pos, obstacles, N):
    """ Get the N closest points to the robot from the obstacles """
    all_points = []
    for line in obstacles:
        all_points.extend(line)
    
    all_points = np.array(all_points)

    # Build KDTree
    kdtree = KDTree(all_points)
    distances, indices = kdtree.query(robot_pos, k=N)
    
    # return the N closest points as an array of shape (N, 2)
    closest_points = all_points[indices]
    
    return closest_points

class HumanRobotEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self, seed=None, options=None, reward_weights=None):
        super(HumanRobotEnv, self).__init__()

        ############### Configurations ################
        if reward_weights is None:
            reward_weights = {
                "translation": 1.0,
                "rotation": 1.0,
                "velocity": 1.0,
                "collision": 1.0,
                "goal": 1.0,
            }
        
        self.trans_wei = reward_weights["translation"]
        self.rot_wei = reward_weights["rotation"]
        self.vel_wei = reward_weights["velocity"]
        self.col_wei = reward_weights["collision"]
        self.goal_wei = reward_weights["goal"]
        
        # Define Obstacles and parameters here
        self.env_obstacles = [[5, -5, -1.3, -0.3], [7, 10, 5, 2]]
        # self.env_obstacles = []
        self.n_obs_points = 0

        self.env_size = 20.0
        self.agent_radius = [0.35, 0.35]  # Robot and Human radius
        # self.desired_distance = np.float32(np.random.uniform(3.0, 7.0)) # This can be randomly set
        self.desired_distance = 1.0  # Object lenght (the distance the robot needs to maintain from the human)
        self.robot_config = {
            "wheel_radius": 0.1,
            "wheel_base": 0.5,
            "desired_dist": self.desired_distance,
            "robot_radius": self.agent_radius[0],
            "step_width": 1.0,
        }    
        self.human_config = {
            "desired_dist": self.desired_distance,
            "human_radius": self.agent_radius[1],
            "resolution": 10.0,
            "path_to_sim_config": None,
        }  
        
        self.use_obs = False
        self.state_dim = 10 # (x, y, theta, vx, vy) for robot and human
        if self.use_obs:
            self.n_obs_points = 1 # Functions sort of like sensor data from a LIDAR
            self.state_dim = 10 + 2 * self.n_obs_points # (x, y, theta, vx, vy) for robot and human + 2 * N closest obstacle points
             
        ############### Configurations ################
        # Normalized action space
        self.action_scale = np.array([3.0, 3.0])  # Scaling factor to map from [-1, 1] to [-3, 3]
        self.action_shift = np.array([0.0, 0.0])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Normalized observation space (x, y, theta, vx, vy) for robot and human
        self.observation_mean = np.zeros(self.state_dim, dtype=np.float32)
        self.observation_std = np.array(
            [self.env_size, self.env_size, np.pi, 3.0, 3.0, # x, y, theta, vx, vy for robot
             self.env_size, self.env_size, np.pi, 3.0, 3.0] + # x, y, theta, vx, vy for human
            [self.env_size, self.env_size] * self.n_obs_points, # 2 * N closest obstacle points
            dtype=np.float32
        )
        
        obs_low = -np.ones(self.state_dim, dtype=np.float32)
        obs_high = np.ones(self.state_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Initialize the state of the environment
        self.reset()
        
    def normalize_observation(self, state):
        return (state - self.observation_mean) / self.observation_std
    
    def denormalize_observation(self, state):
        return (state * self.observation_std) + self.observation_mean

    def rescale_action(self, action):
        # Use a tanh squashing function to ensure actions stay within valid range
        action = np.tanh(action)
        # Rescale the action to the original action range for wheel velocities
        return action * self.action_scale + self.action_shift

    def step(self, action):

        terminated = False
        truncated = False
        check = False
        human_halt = False

        # Count thingy might be useful for debugging or some other purposes
        if self.count > 1 and self.count % 30 == 0:
            check = True
            
        scaled_action = self.rescale_action(action)
        
        prev_human_pos = self.human.pos.copy()
        prev_robot_pos = self.robot.pos.copy()
        
        perv_robot_theta = self.robot.theta.copy()
        perv_human_theta = self.human.theta.copy()
        
        prev_robot_vx, prev_robot_vy = self.robot.get_velocities()
        prev_human_vx, prev_human_vy = self.human.get_velocities()
        
        prev_robot_state = np.concatenate([prev_robot_pos, [perv_robot_theta], [prev_robot_vx, prev_robot_vy]]).astype(np.float32)
        prev_human_state = np.concatenate([prev_human_pos, [perv_human_theta], [prev_human_vx, prev_human_vy]]).astype(np.float32)

        # Update robot position and orientation based on action
        self.robot.update(scaled_action)

        # Update human position and orientation
        self.human.update()

        # Update the history of positions
        self.robot_pos_history.append(self.robot.pos.copy())
        self.human_pos_history.append(self.human.pos.copy())

        # current agent states
        robot_pos = self.robot.pos
        human_pos = self.human.pos

        robot_theta = self.robot.theta
        human_theta = self.human.theta
        
        robot_vx, robot_vy = self.robot.get_velocities()
        human_vx, human_vy = self.human.get_velocities()
        
        curr_robot_state = np.concatenate([robot_pos, [robot_theta], [robot_vx, robot_vy]]).astype(np.float32)
        curr_human_state = np.concatenate([human_pos, [human_theta], [human_vx, human_vy]]).astype(np.float32)


        # Calculate collision penalty
        collision_penalty, done = self.reward.calculate_collision_penalty(robot_pos)

        # Calculate translation reward
        translation_reward, orientation_reward, dist_to_human, rel_theta, diff_angle = self.reward.calculate_distance_reward(curr_robot_state, prev_robot_state, curr_human_state, prev_human_state)

        # Calculate goal reward
        goal_reward, human_arrive, robot_arrive = self.reward.calculate_goal_reward(robot_pos, prev_robot_pos, human_pos, prev_human_pos)
            
        # Calculate velocity reward
        velocity_reward = self.reward.calculate_velocity_reward(curr_robot_state, curr_human_state)
        
        # Check if robot is out of bounds
        offside, offside_reward = self.reward.is_out_of_bounds(robot_pos)

        # Total reward with weights
        reward = (
            self.trans_wei * translation_reward
            + self.rot_wei * orientation_reward
            + self.col_wei * collision_penalty
            + self.goal_wei * goal_reward
            + self.vel_wei * velocity_reward
            + offside_reward
        )

        # Condition for Termination
        # 1. Human or Robot reaches the goal
        # 2. Collision detected
         # dumaan na sa goal yung human (funny copilot moment)
        if human_arrive or robot_arrive:
            # print("Human Arrived")
            terminated = True # bug: This was set to False, should be True
            

        # Condition for Truncation
        # 1. Robot Going out of bounds (offside)
        # 2. Collision detected (done)
        # 3. Human not moving (human_halt)
        if check:
            human_dist_change = np.linalg.norm(human_pos - prev_human_pos)
            if human_dist_change < 0.01 and human_arrive == False:
                human_halt = True  # this thing gets triggered when the human reaches the goal (meaning the robot stops) which is techincally true, not what i want so need to figurue out a way to fix this
                print("Human not moving")
                
        if done or offside or human_halt:
            truncated = True

        info = {
            "Distance to Human": dist_to_human,
            "collision_penalty": collision_penalty,
            "translation_reward": translation_reward,
            "orientation_reward": orientation_reward,
            "goal_reward": goal_reward,
            "velocity_reward": velocity_reward,
            "Goal Arrived": robot_arrive,
            "Collision detected": done,
            "Action_dim": scaled_action.shape,
        }

        if self.use_obs:
            obstacles = self.human.get_obstacles_as_points()
            closest_obstacle_points = get_obstacle_points(robot_pos, obstacles, self.n_obs_points)
            state = np.concatenate([
                robot_pos, [robot_theta], [robot_vx, robot_vy], 
                human_pos, [human_theta], [human_vx, human_vy],
                closest_obstacle_points.flatten()
            ]).astype(np.float32)
        else:
            state = np.concatenate([
                robot_pos, [robot_theta], [robot_vx, robot_vy], 
                human_pos, [human_theta], [human_vx, human_vy]
            ]).astype(np.float32)
        
        normalized_state = self.normalize_observation(state)
        self.count += 1
        # print("env step count: ", self.count, "check: ", check, "human_halt: ", human_halt)
        return normalized_state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # define reset function
        if seed is not None:
            np.random.seed(seed)

        self.count = 0
        
        # Initialize the human 
        if self.env_obstacles is not None:
            self.human = Human_SF(self.human_config, self.env_size, self.env_obstacles)
            self.robot = RobotDiff(self.robot_config, self.human.initial_state, self.env_obstacles)
        else:
            self.human = Human_SF(self.env_size)
            self.robot = RobotDiff(self.robot_config, self.human.initial_state)   
        
        # Initialize the reward calculator
        # self.reward = RewardCalculator(self.desired_distance, self.agent_radius, self.env_obstacles)
        self.reward = RewardCalculator(self.human, self.robot)
        
        # Need to clean this up later
        human_pos = self.human.pos
        human_theta = self.human.theta
        human_vx, human_vy = self.human.get_velocities()
        robot_pos = self.robot.pos
        robot_theta = self.robot.theta
        robot_vx, robot_vy = self.robot.get_velocities()
          
        # Clearing trajectory history for both agents
        self.robot_pos_history = []
        self.human_pos_history = []
        
        # initialize state as an array of shape self.state_dim
        # if use_obs is True, the state should include the N(self.n_obs_points) closest obstacle points
        # else it should just be robot_pos, robot_theta, robot_vx, robot_vy, human_pos, human_theta, human_vx, human_vy
        if self.use_obs:
            obstacles = self.human.get_obstacles_as_points()
            closest_obstacle_points = get_obstacle_points(robot_pos, obstacles, self.n_obs_points)
            state = np.concatenate([
                robot_pos, [robot_theta], [robot_vx, robot_vy], 
                human_pos, [human_theta], [human_vx, human_vy],
                closest_obstacle_points.flatten()
            ]).astype(np.float32)
        else:
            state = np.concatenate([
                robot_pos, [robot_theta], [robot_vx, robot_vy], 
                human_pos, [human_theta], [human_vx, human_vy]
            ]).astype(np.float32)
        
        normalized_state = self.normalize_observation(state)
        info = {'state_dim': normalized_state.shape}
        return normalized_state, info

    def render(self, mode="human"):
        plt.clf()

        arrow_size = 1.0
        robot_radius = self.agent_radius[0]
        human_radius = self.agent_radius[1]

        robot_positions = np.array(self.robot_pos_history)
        human_positions = np.array(self.human_pos_history)
        if len(robot_positions) > 1:
            plt.plot(
                robot_positions[:, 0],
                robot_positions[:, 1],
                "r-",
                label="Robot Trajectory",
            )
        if len(human_positions) > 1:
            plt.plot(
                human_positions[:, 0],
                human_positions[:, 1],
                "b-",
                label="Human Trajectory",
            )
            
        # Plot the initial pose of the robot and human
        plt.arrow(
            self.robot.initial_state[0],
            self.robot.initial_state[1],
            0.5 * np.cos(self.robot.initial_state[2]),
            0.5 * np.sin(self.robot.initial_state[2]),
            head_width=arrow_size,
            head_length=arrow_size,
            fc="g",
            ec="g",      
        )
        
        plt.arrow(
            self.human.initial_state[0],
            self.human.initial_state[1],
            0.5 * np.cos(self.human.initial_state[2]),
            0.5 * np.sin(self.human.initial_state[2]),
            head_width=arrow_size,
            head_length=arrow_size,
            fc="c",
            ec="c",      
        )

        # Plot the robot with an arrow indicating orientation
        plt.arrow(
            self.robot.pos[0],
            self.robot.pos[1],
            0.5 * np.cos(self.robot.theta),
            0.5 * np.sin(self.robot.theta),
            head_width=arrow_size,
            head_length=arrow_size,
            fc="r",
            ec="r",
        )
        # Plot the human with an arrow indicating orientation
        plt.arrow(
            self.human.pos[0],
            self.human.pos[1],
            0.5 * np.cos(self.human.theta),
            0.5 * np.sin(self.human.theta),
            head_width=arrow_size,
            head_length=arrow_size,
            fc="b",
            ec="b",
        )
        # Plot the distance between the robot and the human as a line
        plt.plot(
            [self.robot.pos[0], self.human.pos[0]],
            [self.robot.pos[1], self.human.pos[1]],
            "g--",
        )

        # Draw a circle around the robot to indicate its radius
        robot_circle = plt.Circle(
            (self.robot.pos[0], self.robot.pos[1]),
            robot_radius,
            color="blue",
            fill=False,
        )
        plt.gca().add_patch(robot_circle)
        
        # Draw a circle around the robot to indicate its radius
        human_circle = plt.Circle(
            (self.human.pos[0], self.human.pos[1]),
            human_radius,
            color="red",
            fill=False,
        )
        plt.gca().add_patch(human_circle)


        # Draw the obstacles
        for obstacle in self.human.get_obstacles_as_points():
            plt.plot(
                obstacle[:, 0], obstacle[:, 1], "-o", color="black", markersize=2.5
            )

        # plot the goal position
        human_goal_pos = self.human.get_goal_pos()
        plt.plot(human_goal_pos[0], human_goal_pos[1], "gx", label="Human goal Position")
        robot_goal_pos = self.robot.get_goal_pos()
        plt.plot(robot_goal_pos[0], robot_goal_pos[1], "bx", label="Robot goal Position")

        plt.xlim(-self.env_size, self.env_size)
        plt.ylim(-self.env_size, self.env_size)
        plt.legend()
        plt.pause(0.01)

    def close(self):
        plt.close()
