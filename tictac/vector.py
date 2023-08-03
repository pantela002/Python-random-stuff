import numpy as np
import matplotlib.pyplot as plt

def calculate_collision(ball_A_pos, ball_B_pos, ball_B_vector):
    # Unpack the positions and vector components
    x1, y1 = ball_A_pos
    x2, y2 = ball_B_pos
    dx2, dy2 = ball_B_vector

    # Calculate the relative position and velocity vectors
    rel_pos = np.array([x2 - x1, y2 - y1])
    rel_vel = np.array([dx2, dy2])

    # Calculate the normal vector between the two balls
    norm_vector = rel_pos / np.linalg.norm(rel_pos)

    # Calculate the velocity component parallel to the normal vector
    vel_parallel = np.dot(rel_vel, norm_vector) * norm_vector

    # Calculate the velocity component perpendicular to the normal vector
    vel_perpendicular = rel_vel - vel_parallel

    # Swap the velocity components for ball B
    new_vel_parallel = -vel_parallel
    new_vel_perpendicular = vel_perpendicular

    # Calculate the new vector of motion for ball B
    new_ball_B_vector = new_vel_parallel + new_vel_perpendicular

    return new_ball_B_vector.tolist()


# Example usage
ball_A_pos = [1, 6]
ball_B_pos = [8, -1]
ball_B_vector = [1, 1]

# Simulation parameters
num_steps = 100  # Number of simulation steps
dt = 0.1  # Time step size

# Lists to store the ball positions
ball_A_positions = [ball_A_pos]
ball_B_positions = [ball_B_pos]

# Perform the simulation
for step in range(num_steps):
    # Calculate the new vector of motion for ball B
    new_ball_B_vector = calculate_collision(ball_A_pos, ball_B_pos, ball_B_vector)

    # Update the positions of the balls
    ball_A_pos = [ball_A_pos[0] + ball_B_vector[0] * dt, ball_A_pos[1] + ball_B_vector[1] * dt]
    ball_B_pos = [ball_B_pos[0] + new_ball_B_vector[0] * dt, ball_B_pos[1] + new_ball_B_vector[1] * dt]

    # Store the updated positions
    ball_A_positions.append(ball_A_pos)
    ball_B_positions.append(ball_B_pos)

    # Update the ball B vector for the next iteration
    ball_B_vector = new_ball_B_vector

# Convert the ball positions to NumPy arrays for plotting
ball_A_positions = np.array(ball_A_positions)
ball_B_positions = np.array(ball_B_positions)

# Plot the trajectory of the balls
plt.plot(ball_A_positions[:, 0], ball_A_positions[:, 1], label='Ball A')
plt.plot(ball_B_positions[:, 0], ball_B_positions[:, 1], label='Ball B')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ball Collision Simulation')
plt.legend()
plt.grid(True)
plt.show()