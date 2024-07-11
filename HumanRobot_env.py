import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from Human import Human
from Human_SF import Human_SF
from Robot import Robot
from RobotDiff import RobotDiff
from Rewards import RewardCalculator


class HumanRobotEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self, seed=None, options=None, reward_weights=None):
        super(HumanRobotEnv, self).__init__()

        if reward_weights is None:
            reward_weights = {
                "translation": 1.0,
                "rotation": 1.0,
                "velocity": 1.0,
                "collision": 1.0,
                "goal": 1.0,
            }

        print("Reward Weights from env script: ", reward_weights)
        
        self.trans_wei = reward_weights["translation"]
        self.rot_wei = reward_weights["rotation"]
        self.vel_wei = reward_weights["velocity"]
        self.col_wei = reward_weights["collision"]
        self.goal_wei = reward_weights["goal"]
        # add obstacles here
        self.obs = [[5, -5, -1.3, -0.3], [7, 10, 5, 2]]

        self.env_size = 20.0
        self.agent_radius = [0.35, 0.35]  # Robot and Human radius
        # self.desired_distance = np.float32(np.random.uniform(3.0, 7.0)) # This can be randomly set
        self.desired_distance = 1.0  # Object lenght (the distance the robot needs to maintain from the human)
        self.robot_config = {
            "wheel_radius": 0.1,
            "wheel_base": 0.5,
            "desired_dist": self.desired_distance,
            "robot_radius": self.agent_radius[0],
        }

        # defining action space (continuous or discrete) the actions that the agent can take is in the form of wheel velocity (v_left, v_right)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]), high=np.array([5.0, 5.0]), dtype=np.float32
        )

        # defining observation space
        # [robot_x, robot_y, robot_theta, human_x, human_y, human_theta]
        obs_low = np.array(
            [
                -self.env_size,
                -self.env_size,
                -np.pi,
                -self.env_size,
                -self.env_size,
                -np.pi,
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [self.env_size, self.env_size, np.pi, self.env_size, self.env_size, np.pi],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Initialize the state of the environment
        self.reset()

    def step(self, action):

        terminated = False
        truncated = False
        check = False

        # Count thingy might be useful for debugging or some other purposes
        # if self.count > 1 and self.count % 10 == 0:
        #     prev_human_pos = self.human.pos.copy()
        #     check = True
        
        prev_human_pos = self.human.pos.copy()
        prev_robot_pos = self.robot.pos.copy()
        
        perv_robot_theta = self.robot.theta.copy()
        perv_human_theta = self.human.theta.copy()
        
        prev_robot_vx, prev_robot_vy = self.robot.get_velocities()
        prev_human_vx, prev_human_vy = self.human.get_velocities()
        
        prev_robot_state = np.concatenate([prev_robot_pos, [perv_robot_theta], [prev_robot_vx, prev_robot_vy]]).astype(np.float32)
        prev_human_state = np.concatenate([prev_human_pos, [perv_human_theta], [prev_human_vx, prev_human_vy]]).astype(np.float32)

        # Update robot position and orientation based on action
        self.robot.update(action)

        # Update human position and orientation
        self.human.update()

        # Update the history of positions
        self.robot_pos_history.append(self.robot.pos.copy())
        self.human_pos_history.append(self.human.pos.copy())

        # Get obstacles and goal position
        obstacles = self.human.get_obstacles()
        goal_pos = self.human.get_goal_pos()
        
        # current agent states
        robot_pos = self.robot.pos
        human_pos = self.human.pos

        robot_theta = self.robot.theta
        human_theta = self.human.theta
        
        robot_vx, robot_vy = self.robot.get_velocities()
        human_vx, human_vy = self.human.get_velocities()
        
        curr_robot_state = np.concatenate([robot_pos, [robot_theta], [robot_vx, robot_vy]]).astype(np.float32)
        curr_human_state = np.concatenate([human_pos, [human_theta], [human_vx, human_vy]]).astype(np.float32)

        # distance to the human
        distance_to_human = np.linalg.norm(robot_pos - human_pos)
        # orientation difference to the human
        orientation_difference = np.abs(robot_theta - human_theta)

        # Calculate collision penalty
        collision_penalty = self.reward.calculate_collision_penalty(robot_pos)

        # Calculate translation reward
        translation_reward = self.reward.calculate_distance_reward(curr_robot_state, prev_robot_state, curr_human_state, prev_human_state)

        # Calculate goal reward
        goal_reward = self.reward.calculate_goal_reward(robot_pos, prev_robot_pos, goal_pos)
        
        # Calculate rotation reward
        rotation_reward = -orientation_difference
        
        # Calculate velocity reward
        velocity_reward = self.reward.calculate_velocity_reward(curr_robot_state, curr_human_state)

        # Total reward with weights
        reward = (
            self.trans_wei * translation_reward
            + self.rot_wei * rotation_reward
            + self.col_wei * collision_penalty
            + self.goal_wei * goal_reward
            + self.vel_wei * velocity_reward
        )

        # Termination conditions
        # if check is True and np.array_equal(human_pos, prev_human_pos):
        #     terminated = True

        # Terminated when Human reached goal position
        # if np.array_equal(human_pos, goal_pos):
        #     terminated = True
        if np.linalg.norm(human_pos - goal_pos) < 0.5:
            terminated = True

        info = {
            # 'distance_to_human': distance_to_human,
            # 'orientation_difference': orientation_difference,
            "collision_penalty": collision_penalty,
            "translation_reward": translation_reward,
            "rotation_reward": rotation_reward,
            "goal_reward": goal_reward,
            'velocity_reward': velocity_reward,
        }

        state = np.concatenate(
            [robot_pos, [self.robot.theta], human_pos, [self.human.theta]]
        ).astype(np.float32)

        # self.count += 1
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # define reset function
        if seed is not None:
            np.random.seed(seed)

        self.count = 0
        # self.human = Human()
        self.human = Human_SF(self.obs)

        # self.robot = Robot(obj_lenght=self.desired_distance, human_pos=self.human.pos)
        self.robot = RobotDiff(self.robot_config, human_pos=self.human.pos)
        self.reward = RewardCalculator(self.desired_distance, self.agent_radius, self.obs)

        self.robot_pos_history = []
        self.human_pos_history = []

        state = np.concatenate(
            [self.robot.pos, [self.robot.theta], self.human.pos, [self.human.theta]]
        ).astype(np.float32)

        info = {}
        return state, info

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

        # Draw the obstacles
        for obstacle in self.human.get_obstacles():
            plt.plot(
                obstacle[:, 0], obstacle[:, 1], "-o", color="black", markersize=2.5
            )

        # plot the goal position
        goal_pos = self.human.get_goal_pos()
        plt.plot(goal_pos[0], goal_pos[1], "gx", label="Goal Position")

        plt.xlim(-self.env_size, self.env_size)
        plt.ylim(-self.env_size, self.env_size)
        plt.legend()
        plt.pause(0.01)

    def close(self):
        plt.close()
