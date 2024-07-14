import numpy as np

def on_segment(p, q, r):
    '''Given three colinear points p, q, r, the function checks if 
    point q lies on line segment "pr"
    '''
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    '''Find orientation of ordered triplet (p, q, r).
    The function returns following values
    0 --> p, q and r are colinear
    1 --> Clockwise
    2 --> Counterclockwise
    '''

    val = ((q[1] - p[1]) * (r[0] - q[0]) - 
            (q[0] - p[0]) * (r[1] - q[1]))
    if val == 0:
        return 0  # colinear
    elif val > 0:
        return 1   # clockwise
    else:
        return 2  # counter-clockwise

def do_intersect(p1, q1, p2, q2):
    '''Main function to check whether the closed line segments p1 - q1 and p2 
       - q2 intersect'''
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2 and o3 != o4):
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 and on_segment(p1, p2, q1)):
        return True

    # p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 and on_segment(p1, q2, q1)):
        return True

    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 and on_segment(p2, p1, q2)):
        return True

    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 and on_segment(p2, q1, q2)):
        return True

    return False # Doesn't fall in any of the above cases

class RobotDiff:
    def __init__(self, config_file, human_state, obstacles=None):
        if config_file is None:
            raise ValueError("Robot Configuration file not found")

        self.wheel_radius = config_file["wheel_radius"]# Radius of the wheels
        self.wheel_base = config_file["wheel_base"]     # Distance between the wheels
        self.desired_dist = config_file["desired_dist"]     
        self.robot_radius = config_file["robot_radius"]  # Radius of the robot
        self.step_width = config_file["step_width"]      # Step width
        self.dist_away = self.desired_dist + self.robot_radius
        self.env_obstacles = obstacles if obstacles is not None else None
        self.reset(human_state)
        
    def update(self, action):
        """This is a discrete non-linear motion model for a differential drive robot has two wheels, each of which can be controlled independently. 
        The robot's movement is determined by the velocities of these wheels."""
        # Action is now [v_left, v_right] - velocities of left and right wheels
        v_left, v_right = action
        # print("v_left, v_right:", v_left, v_right)
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
        self.vy = 0.0 # linear_vel * np.sin(self.theta)
        
        # Store linear and angular velocities for potential use elsewhere
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel
        # print("vx, vy:", self.vx, self.vy)
        
        self.pos_history.append(self.pos.copy())
        
    def reset(self, human_state):
        # self.theta = np.float32(np.random.uniform(-np.pi, np.pi))
        human_pos = human_state[0:2]
        human_theta = human_state[2]
        human_vx, human_vy = human_state[3:5]
        human_goal_pos = human_state[5:7]
        
        self.pos_history = [] # Store the robot's position history
        
        self.init_pos = self.set_valid_pos(human_pos)        
        self.pos = self.init_pos.copy()
        self.theta = human_theta
        self.vx = 0.0
        self.vy = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.goal_pos = self.set_valid_pos(human_goal_pos)
        
        self.pos_history.append(self.pos.copy())
        self.initial_state = np.concatenate([self.pos, [self.theta], [self.vx, self.vy], self.goal_pos])

    def get_velocities(self):
        return self.vx, self.vy # This was supposed to return vx and vy
    
    def get_goal_pos(self):
        return self.goal_pos
    
    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def check_arrive(self, curr_robot_pos):
        return True if np.linalg.norm(curr_robot_pos - self.goal_pos) < 0.5 else False  
    
    def set_valid_pos(self, human_goal_pos):
        '''Sets the goal position for the robot
            input: goal_pos: np.array([x, y])
            output: valid_goal: np.array([x, y])'''
            
        # imagine the goal point(x,y) is the center of a circle with radius self.dist_away,
        # the robot goal is the point on the circumference of the circle
        # i want to sample a n random point from the circumference of the circle and store it as a queue of points
        # maybe a while loop that keeps checking the validity of the point in the queue
        # then check if the line segment formed by the human's goal and the robot's goal intersects with any of the obstacles using is_valid_goal function
        # if it does, pop the point from the queue and sample a new point
        # if it doesn't, return the point as the robot's goal
        queue = []
        max_attempts = 100
        valid_goal = None
        attempts = 0
        
            
        def sample_random_point_on_circle(center, radius, n_points=1):
            """Sample n random points on the circumference of a circle with given center and radius."""
            angles = np.random.uniform(0, 2 * np.pi, n_points)
            points = np.zeros((n_points, 2))
            points[:, 0] = center[0] + radius * np.cos(angles)
            points[:, 1] = center[1] + radius * np.sin(angles)
            return points.tolist()

        while attempts < max_attempts:
            if queue == []:
                queue = sample_random_point_on_circle(human_goal_pos, self.dist_away, n_points=10)
                
            robot_goal_pos = queue.pop(0)
            
            if self.is_valid_goal(human_goal_pos, robot_goal_pos):
                valid_goal = robot_goal_pos
                break
            
            attempts += 1
            
        if valid_goal is None:
            raise ValueError("No valid goal found")
        
        return np.array(valid_goal, dtype=np.float32)
    
    def is_valid_goal(self, human_goal_pos, robot_goal_pos):
        '''Checks if the goal is valid (A valid goal is one that is not within the obstacle space)
            input: goal_pos: np.array([x, y])
            output: True if the goal is valid, False otherwise
            obstacles: list of obstacles in the environment, each obstacle has the form [xmin, xmax, ymin, ymax]'''
        
        # idea is to check if the line segment formed by the human's goal and the robot's goal intersects with any of the obstacles which is also represented as lines
        if self.env_obstacles is not None: # only check if there are obstacles otherwise return True
            for obstacle in self.env_obstacles :
                startx, endx, starty, endy = obstacle
                obstacle_line = [(startx, starty), (endx, endy)]
                
                if do_intersect(human_goal_pos, robot_goal_pos, obstacle_line[0], obstacle_line[1]):
                    return False
                
        return True
        

