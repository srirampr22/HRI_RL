import numpy as np
import math
from math import pi
from scipy.spatial import KDTree

class RewardCalculator:
    def __init__(self, human, robot):
        self.robot = robot
        self.human = human
        self.obstacles = self.human.get_obstacles_as_points()
        self.robot_radius = self.robot.robot_radius
        self.human_radius = self.human.human_radius
        self.desired_distance = self.robot.desired_dist
        self.dist_progressive_reward = 0.0
        self.env_size = self.human.env_size
        
        # self.agent_radius = agent_radius

    def calculate_collision_penalty(self, robot_pos):
        collision_penalty = 0
        robot_radius = self.robot_radius
        done = False
        
        if self.obstacles is not None:
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

            closest_obs_point = get_obstacle_points(robot_pos, self.obstacles, 1)
            for point in closest_obs_point:
                distance = np.linalg.norm(robot_pos - point)
                if distance < robot_radius:
                    collision_penalty = -100
                    done = True
                    break

        return collision_penalty, done
    
    def calculate_velocity_reward(self, curr_robot_state, curr_human_state, velocity_threshold=0.2):
        # Extract velocities
        robot_vx, robot_vy = curr_robot_state[3:5]
        human_vx, human_vy = curr_human_state[3:5]

        # Calculate speeds
        robot_speed = np.sqrt(robot_vx**2 + robot_vy**2)
        human_speed = np.sqrt(human_vx**2 + human_vy**2)

        # Calculate velocity difference
        velocity_diff = abs(robot_speed - human_speed)
        # print("velocity_diff: ", velocity_diff)

        # Reward for matching human's velocity
        if velocity_diff <= velocity_threshold:
            vel_matching_reward = 100  # Positive reward for closely matching human velocity
        else:
            vel_matching_reward = -10 * velocity_diff  # Penalty for velocity difference

        # Calculate direction similarity
        human_direction = np.array([human_vx, human_vy]) / (human_speed + 1e-6)  # Avoid division by zero
        robot_direction = np.array([robot_vx, robot_vy]) / (robot_speed + 1e-6)  # Avoid division by zero
        direction_similarity = np.dot(human_direction, robot_direction)
        direction_reward = 10 * direction_similarity  # Reward for moving in the same direction

        # Combine rewards
        total_reward = vel_matching_reward 

        return total_reward

    def calculate_goal_reward(self, robot_pos, prev_robot_pos, human_pos, prev_human_pos):

        direction_wei = 1
        alpha = 1
        beta = 1
        human_arrive = self.human.check_arrive(human_pos)
        robot_arrive = self.robot.check_arrive(robot_pos)
        robot_goal_pos = self.robot.get_goal_pos()
        current_distance_to_goal = np.linalg.norm(robot_pos - robot_goal_pos)
        previous_distance_to_goal = np.linalg.norm(prev_robot_pos - robot_goal_pos)

        # # Calculate the change in distance
        # distance_change = previous_distance_to_goal - current_distance_to_goal
        # # print("current distance to goal: ", current_distance_to_goal)
        base_reward = 0

        # # Base reward
        # if current_distance_to_goal <= self.desired_distance:
        #     base_reward = 120  # Large positive reward for reaching goal area
        #     arrive = True
        # else:
        #     base_reward = 0

        # # Progressive reward based on distance change
        # if distance_change > 0:  # Robot got closer to the goal
        #     progressive_reward = (
        #         alpha * distance_change
        #     )  # Positive reward proportional to improvement
        # else:  # Robot moved away from the goal
        #     progressive_reward = (
        #         beta * distance_change
        #     )  # Larger negative reward proportional to regression

        # # Goal Diretion Reward
        # goal_direction_vector = goal_pos - prev_robot_pos
        # goal_direction_vector = goal_direction_vector / np.linalg.norm(
        #     goal_direction_vector
        # )  # Normalize
        # robot_movement_vector = robot_pos - prev_robot_pos
        # robot_movement_vector = robot_movement_vector / (
        #     np.linalg.norm(robot_movement_vector) + 1e-6
        # )  # Normalize and avoid division by zero

        # direction_similarity = np.dot(goal_direction_vector, robot_movement_vector)
        # goal_direction_reward = (
        #     direction_wei * direction_similarity
        # )  # Positive reward for moving in the right direction, (-1, 1)

        if robot_arrive and human_arrive:
            base_reward = 1000
        elif robot_arrive and human_arrive == False:
            base_reward = 500
        elif robot_arrive == False and human_arrive:
            base_reward = 0
        
        
        return base_reward, human_arrive, robot_arrive 

    def calculate_distance_reward(
        self, curr_robot_state, prev_robot_state, curr_human_state, prev_human_state
    ):
        desired_distance = self.desired_distance

        # Extract current positions and velocities
        curr_robot_pos = curr_robot_state[:2]
        curr_robot_theta = curr_robot_state[2]
        curr_robot_vx, curr_robot_vy = curr_robot_state[3:5]

        curr_human_pos = curr_human_state[:2]
        curr_human_theta = curr_human_state[2]
        curr_human_vx, curr_human_vy = curr_human_state[3:5]
        
        prev_robot_pos = prev_robot_state[:2]
        prev_human_pos = prev_human_state[:2]

        # Calculate current distance between robot and human
        current_distance = np.linalg.norm(curr_robot_pos - curr_human_pos)
        prev_distance = np.linalg.norm(prev_robot_pos - prev_human_pos)
        
        # rel_diff = abs(current_distance - prev_distance) 
        # if current_distance > prev_distance: 
        #     self.dist_progressive_reward -= 10 * rel_diff
        # else:
        #     self.dist_progressive_reward += 10 * rel_diff
            
        distance_rate = (prev_distance - current_distance)
        reward = 500 * distance_rate
            
        rel_dis_x = curr_robot_pos[0] - curr_human_pos[0]
        rel_dis_y = curr_robot_pos[1] - curr_human_pos[1]
            
        if rel_dis_x > 0 and rel_dis_y > 0:
            rel_theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            rel_theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            rel_theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            rel_theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            rel_theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            rel_theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            rel_theta = 0
        else:
            rel_theta = math.pi
            
        diff_angle = curr_robot_theta - rel_theta
        diff_angle = self.wrap_to_pi(diff_angle) # It indicates how far off the robot's current orientation is from the direction it needs to go to reach the target.
        
        orientation_difference = self.wrap_to_pi(abs(curr_robot_theta - curr_human_theta))
        orientation_reward = -orientation_difference 
        
        trans_reward = reward # self.dist_progressive_reward

        return trans_reward, orientation_reward, current_distance, rel_theta, diff_angle
    
    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def is_out_of_bounds(self, position):
        """
        Check if the robot is out of the bounds of the environment.

        Args:
            position (np.ndarray): The position of the robot as a numpy array [x, y].

        Returns:
            bool: True if the robot is out of bounds, False otherwise.
        """
        x, y = position
        reward = 0
        if x < -self.env_size or x > self.env_size or y < -self.env_size or y > self.env_size:
            reward = -100
            return True, reward
        return False, reward
