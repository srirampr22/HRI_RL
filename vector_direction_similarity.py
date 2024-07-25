import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# def compute_direction_vector(point1, point2):
#     return np.array(point2) - np.array(point1)

# def direction_similarity(points_set1, points_set2):
#     vectors_set1 = [compute_direction_vector(p1, p2) for p1, p2 in zip(points_set1[:-1], points_set1[1:])]
#     vectors_set2 = [compute_direction_vector(p1, p2) for p1, p2 in zip(points_set2[:-1], points_set2[1:])]
    
#     similarities = cosine_similarity(vectors_set1, vectors_set2)
#     avg_similarity = np.mean(similarities)
    
#     return avg_similarity, vectors_set1, vectors_set2

# def plot_vectors(points_set1, points_set2, vectors_set1, vectors_set2):
#     plt.figure(figsize=(10, 6))
    
#     # Plot points set 1
#     points_set1 = np.array(points_set1)
#     plt.plot(points_set1[:, 0], points_set1[:, 1], 'bo-', label='Points Set 1')
    
#     # Plot direction vectors for set 1
#     for i, vector in enumerate(vectors_set1):
#         plt.arrow(points_set1[i, 0], points_set1[i, 1], vector[0], vector[1], 
#                   head_width=0.2, head_length=0.3, fc='blue', ec='blue')
    
#     # Plot points set 2
#     points_set2 = np.array(points_set2)
#     plt.plot(points_set2[:, 0], points_set2[:, 1], 'ro-', label='Points Set 2')
    
#     # Plot direction vectors for set 2
#     for i, vector in enumerate(vectors_set2):
#         plt.arrow(points_set2[i, 0], points_set2[i, 1], vector[0], vector[1], 
#                   head_width=0.2, head_length=0.3, fc='red', ec='red')
    
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Direction Vectors and Points')
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Example usage
# points_set1 = [(1, 2), (3, -1)]
# points_set2 = [(2, 2), (4, 2)]

# similarity, vectors_set1, vectors_set2 = direction_similarity(points_set1, points_set2)
# print(f'Direction Similarity: {similarity}')

# plot_vectors(points_set1, points_set2, vectors_set1, vectors_set2)



def compute_wheel_velocities(vx, vy, wheel_radius, wheel_base, robot_radius):
    # Compute linear and angular velocities
    v = np.sqrt(vx**2 + vy**2)
    omega = vy / robot_radius  # This might be zero if vy is zero

    # Compute left and right wheel velocities
    v_left = (2 * v - wheel_base * omega) / (2 * wheel_radius)
    v_right = (2 * v + wheel_base * omega) / (2 * wheel_radius)

    return v_left, v_right

# Example usage
vx = 0.85  # m/s
vy = 0.0   # m/s
wheel_radius = 0.1  # meters
wheel_base = 0.5    # meters
robot_radius = 0.35 # meters
print("Wheel Velocities:", compute_wheel_velocities(vx, vy, wheel_radius, wheel_base, robot_radius))