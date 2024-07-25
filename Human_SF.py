import numpy as np
import pysocialforce as psf
from scipy.spatial import KDTree


class Human_SF:
    def __init__(self, config_file, env_size, env_obstacles=None): 
        self.desired_dist = config_file["desired_dist"]     
        self.human_radius = config_file["human_radius"]  # Radius of the robot
        self.resolution = config_file["resolution"]      # Resolution of the obstacles 
        self.path_to_sim_config = config_file["path_to_sim_config"]
        self.obstacle_lines = env_obstacles if env_obstacles is not None else None
        self.obstacle_points = self.obstacles_as_points(env_obstacles) if env_obstacles is not None else None
        self.env_size = env_size
        self.reset()
        

    def update(self):
        # Use the simulator to update the human's position
        prev_pos = self.pos
        self.sim.step()

        pedestrian_states, group_states = (
            self.sim.get_states()
        )  # Get the state of the pedestrians and groups
        latest_state = pedestrian_states[-1][0]

        # Update the position and orientation
        self.pos = np.array([latest_state[0], latest_state[1]], dtype=np.float32)
        self.theta = np.arctan2(latest_state[3], latest_state[2])
        self.theta = self.wrap_to_pi(self.theta)
        self.vx = latest_state[2]
        self.vy = latest_state[3]
        
        curr_pos = self.pos
        vx_, vy_ = self.compute_velocity_components(prev_pos, curr_pos, self.theta, 1.0)
        
        # print("vx, vy:", vx_, vy_)
        self.pos_history.append(self.pos.copy())

    def reset(self):
        # Set the initial position of the human within the range [-10.0, 10.0]
        cntrl_limit = self.env_size - 5
        init_pos = self.set_init_pos()
        # init_pos = np.array([-10.0, -10.0], dtype=np.float32)
        init_theta = np.float32(np.random.uniform(-np.pi, np.pi))
        vx = np.float32(0.5)  # Initial velocity x-component
        vy = np.float32(0.5)  # Initial velocity y-component
        goal_pos = self.set_goal(init_pos)
        # goal_pos = np.array([18.0, 18.0], dtype=np.float32)
        self.pos_history = []  # Store the human's position history
        self.init_sim_state = np.array(
            [
                [init_pos[0], init_pos[1], vx, vy, goal_pos[0], goal_pos[1]],
                # [-10.0, -10.0, 0.0, 0.5, 18.0, 18.0],
            ]
        )

        self.sim = psf.Simulator(
            self.init_sim_state,
            groups=None,
            obstacles=self.obstacle_lines,
            config_file="/home/sriram/gym_play/PySocialForce/examples/example.toml",
        )

        self.pos = np.array(
            [self.init_sim_state[0][0], self.init_sim_state[0][1]], dtype=np.float32
        )
        
        self.pos_history.append(self.pos.copy())
        # self.theta = np.arctan2(self.init_sim_state[0][3], self.init_sim_state[0][2])
        self.theta = self.wrap_to_pi(init_theta)
        self.vx = self.init_sim_state[0][2]
        self.vy = self.init_sim_state[0][3]
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        
        self.initial_state = np.concatenate([self.pos, [self.theta], [self.vx, self.vy], self.goal_pos])

    def get_obstacles_as_points(self):
        # print("Obstacle Points:", self.obstacle_points)  # Debug print
        return self.obstacle_points if self.obstacle_points is not None else None
    
    def get_obstacles_as_lines(self):
        return self.obstacle_lines if self.obstacle_lines is not None else None

    def get_goal_pos(self):
        # print("Initial State:", self.initial_state)  # Debug print
        # print("Length of Initial State:", len(self.initial_state))  # Debug print
        return np.array(self.goal_pos, dtype=np.float32)

    def get_velocities(self):
        return self.vx, self.vy
    
    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def check_arrive(self, curr_human_pos):
        return True if np.linalg.norm(curr_human_pos - self.initial_state[5:7]) < 0.5 else False
    
    def set_goal(self, init_pos):
        max_attempts = 100
        attempts = 0
        valid_goal = None
        queue = []
        
        while attempts < max_attempts:
            if queue == []:
                queue = self.sample_goal_points(init_pos, n_points=10)
                
            goal_pos = queue.pop(0)
    
            if self.is_valid_point(goal_pos):
                valid_goal = goal_pos
                break
            
            attempts += 1
            
        if valid_goal is None:
            raise ValueError("No valid human goal found")
        
        return np.array(valid_goal, dtype=np.float32)
    
    def set_init_pos(self):
        max_attempts = 100
        attempts = 0
        valid_init_pos = None
        queue = []
        
        while attempts < max_attempts:
            if queue == []:
                queue = self.sample_points(n_points=10)

            init_pos = queue.pop(0)
            
            if self.is_valid_point(init_pos):
                valid_init_pos = init_pos
                break
            
            attempts += 1
            
        if valid_init_pos is None:
            raise ValueError("No valid initial human position found")
        
        return np.array(valid_init_pos, dtype=np.float32)
        
    def is_valid_point(self, goal_pos):
        '''Checks if the goal point is valid (A valid goal is one that is not within the obstacle space)
            input: goal_pos: np.array([x, y])
            output: True if the goal is valid, False otherwise
            obstacles: list of obstacles in the environment, each obstacle has the form [xmin, xmax, ymin, ymax]'''
        
        
        def get_obstacle_points(pos, obstacles, N=1):
                """ Get the N closest points to the agent from the obstacles """
                all_points = []
                obstacles = self.get_obstacles_as_points()
                for line in obstacles:
                    all_points.extend(line)
                
                all_points = np.array(all_points)

                # Build KDTree
                kdtree = KDTree(all_points)
                distances, indices = kdtree.query(pos, k=N)
                
                # return the N closest points as an array of shape (N, 2)
                closest_points = all_points[indices]
                
                return closest_points
            
        if self.obstacle_points is not None: # only check if there are obstacles otherwise return True
            print("Obstacle Points:", self.obstacle_points)
            obstacles = self.get_obstacles_as_points() 
            closest_obs_point = get_obstacle_points(goal_pos, obstacles, 1)
            for point in closest_obs_point:
                distance = np.linalg.norm(goal_pos - point)
                dist_away = self.human_radius * 2 + self.desired_dist
                if distance < dist_away:
                    return False
        return True
    
    def sample_goal_points(self, init_pos, n_points=1):
        """Sample n random points within the environment ensuring they are in different quadrants."""
        goal_limit = self.env_size - 5
        points = []

        for _ in range(n_points):
            while True:
                x = np.random.uniform(-goal_limit, goal_limit)
                y = np.random.uniform(-goal_limit, goal_limit)
                point = np.array([x, y])

                if self.is_in_different_quadrant(point, init_pos):
                    points.append(point)
                    break

        return points

    def is_in_different_quadrant(self, point, initial_pos):
        """Check if the point is in a different quadrant than the initial position."""
        def get_quadrant(pos):
            if pos[0] >= 0 and pos[1] >= 0:
                return 1
            elif pos[0] < 0 and pos[1] >= 0:
                return 2
            elif pos[0] < 0 and pos[1] < 0:
                return 3
            else:
                return 4

        initial_quadrant = get_quadrant(initial_pos)
        point_quadrant = get_quadrant(point)

        return point_quadrant != initial_quadrant
    
    def sample_points(self, n_points=1):
        """Sample n random points within the environment."""
        points = []
        pos_limit = self.env_size - 5

        for _ in range(n_points):
            x = np.float32(np.random.uniform(-pos_limit, pos_limit))
            y = np.float32(np.random.uniform(-pos_limit, pos_limit))
            points.append(np.array([x, y]))
        return points
    
    def obstacles_as_points(self, obstacles):
        """Input an list of (startx, endx, starty, endy) as start and end of a line"""
        obstacle_points = []
        if obstacles is None:
            print("No obstacles")
            obstacle_points = []
        else:
            obstacle_points = []
            for startx, endx, starty, endy in obstacles:
                samples = int(np.linalg.norm((startx - endx, starty - endy)) * self.resolution)
                line = np.array(
                    list(
                        zip(np.linspace(startx, endx, samples), np.linspace(starty, endy, samples))
                    )
                )
                obstacle_points.append(line)
                
        return obstacle_points
    
    def compute_velocity_components(self, prev_pos, curr_pos, theta, delta_t):
        """
        Compute the velocity components vx and vy given previous and current positions,
        orientation theta, and time difference delta_t.

        :param prev_pos: tuple, previous position (x1, y1)
        :param curr_pos: tuple, current position (x2, y2)
        :param theta: float, orientation in radians
        :param delta_t: float, time difference between the positions
        :return: tuple, velocity components (vx, vy)
        """
        x1, y1 = prev_pos
        x2, y2 = curr_pos
        
        # Calculate change in position
        delta_x = x2 - x1
        delta_y = y2 - y1
        
        # Compute velocity components
        vx = delta_x / delta_t
        vy = delta_y / delta_t
        
        return vx, vy
        
            
        
        

        
        
        
        
    
