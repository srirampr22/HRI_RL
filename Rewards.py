import numpy as np


class RewardCalculator:
    def __init__(self, desired_distance=5.0, agent_radius=None, obstacles=None):
        self.desired_distance = desired_distance
        self.agent_radius = agent_radius
        self.obstacles = obstacles if obstacles is not None else []

    def calculate_collision_penalty(self, robot_pos):
        collision_penalty = 0
        robot_radius = self.agent_radius[0]

        # Check collision with obstacles
        for obstacle in self.obstacles:
            x_min, x_max, y_min, y_max = obstacle
            if (
                x_min - robot_radius <= robot_pos[0] <= x_max + robot_radius
                and y_min - robot_radius <= robot_pos[1] <= y_max + robot_radius
            ):
                collision_penalty -= 100  # Large negative reward for collision
                break

        return collision_penalty

    def calculate_velocity_reward(
        self,
        curr_robot_state,
        curr_human_state,
        base_velocity=0.5,
        velocity_threshold=0.2,
    ):

        robot_vx, robot_vy = curr_robot_state[3:5]
        human_vx, human_vy = curr_human_state[3:5]

        # Calculate speeds
        robot_speed = np.sqrt(robot_vx**2 + robot_vy**2)
        human_speed = np.sqrt(human_vx**2 + human_vy**2)

        # Calculate velocity difference
        velocity_diff = abs(robot_speed - human_speed)

        # Base reward for maintaining a certain velocity
        if abs(robot_speed - base_velocity) <= velocity_threshold:
            base_reward = 50  # Positive reward for maintaining base velocity
        else:
            base_reward = -10 * abs(
                robot_speed - base_velocity
            )  # Penalty for deviating from base velocity

        # Reward for matching human's velocity
        if velocity_diff <= velocity_threshold:
            vel_matching_reward = (
                50  # Positive reward for closely matching human velocity
            )
        else:
            vel_matching_reward = -10 * velocity_diff  # Penalty for velocity difference

        # Combine rewards
        total_reward = base_reward + vel_matching_reward

        return total_reward

    def calculate_goal_reward(self, robot_pos, prev_robot_pos, goal_pos):

        direction_wei = 100
        alpha = 10
        beta = 10

        current_distance_to_goal = np.linalg.norm(robot_pos - goal_pos)
        previous_distance_to_goal = np.linalg.norm(prev_robot_pos - goal_pos)

        # Calculate the change in distance
        distance_change = previous_distance_to_goal - current_distance_to_goal

        # Base reward
        if current_distance_to_goal <= self.desired_distance:
            base_reward = 1000  # Large positive reward for reaching goal area
        else:
            base_reward = 0

        # Progressive reward based on distance change
        if distance_change > 0:  # Robot got closer to the goal
            progressive_reward = (
                alpha * distance_change
            )  # Positive reward proportional to improvement
        else:  # Robot moved away from the goal
            progressive_reward = (
                beta * distance_change
            )  # Larger negative reward proportional to regression

        # Goal Diretion Reward
        goal_direction_vector = goal_pos - prev_robot_pos
        goal_direction_vector = goal_direction_vector / np.linalg.norm(
            goal_direction_vector
        )  # Normalize
        robot_movement_vector = robot_pos - prev_robot_pos
        robot_movement_vector = robot_movement_vector / (
            np.linalg.norm(robot_movement_vector) + 1e-6
        )  # Normalize and avoid division by zero

        direction_similarity = np.dot(goal_direction_vector, robot_movement_vector)
        goal_direction_reward = (
            direction_wei * direction_similarity
        )  # Positive reward for moving in the right direction, (-1, 1)

        return base_reward + progressive_reward + goal_direction_reward

    def calculate_distance_reward(
        self, curr_robot_state, prev_robot_state, curr_human_state, prev_human_state
    ):
        desired_distance = self.desired_distance

        # Extract current positions and velocities
        curr_robot_pos = curr_robot_state[:2]
        curr_robot_theta = curr_robot_state[2]
        curr_robot_vx, curr_robot_vy = curr_robot_state[3:5]

        curr_human_pos = curr_human_state[:2]
        curr_human_vx, curr_human_vy = curr_human_state[3:5]

        # Calculate current distance between robot and human
        current_distance = np.linalg.norm(curr_robot_pos - curr_human_pos)

        # Calculate speeds
        human_speed = np.sqrt(curr_human_vx**2 + curr_human_vy**2)
        robot_speed = np.sqrt(curr_robot_vx**2 + curr_robot_vy**2)

        # Base reward for maintaining desired distance
        if (
            abs(current_distance - desired_distance) <= 0.5
        ):  # Within 0.5 units of desired distance
            base_reward = 100
        else:
            base_reward = -10 * abs(current_distance - desired_distance)

        # Keeping up with human movement
        if human_speed > 0:
            # Calculate the relative speed
            relative_speed = robot_speed / human_speed

            if 0.9 <= relative_speed <= 1.1:
                # Robot is matching human's speed well (within 10% margin)
                speed_reward = 50
            elif relative_speed > 1.1:
                # Robot is moving faster than the human
                speed_reward = 20 - 10 * (relative_speed - 1.1)
            else:
                # Robot is moving slower than the human
                speed_reward = 20 - 30 * (0.9 - relative_speed)

            keeping_up_reward = speed_reward
        else:
            # If human isn't moving, reward the robot for staying still
            keeping_up_reward = 20 if robot_speed < 0.1 else -10 * robot_speed

        total_reward = base_reward + keeping_up_reward

        return total_reward
