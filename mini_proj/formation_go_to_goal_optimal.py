"""
Fundamental of Mobile Robot - AUT.710 - 2023
Mini Project - Formations go to goal with dynamic obstacles in QP
Hoang Pham, Nadun Ranasinghe, Dong Le
"""

import cvxopt
import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robots import sim_mobile_robots

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 20 # total simulation duration in seconds
IS_SHOWING_2DVISUALIZATION = True
ROBOT_NUM = 4
CLUSTER_NUM = 1
DYNAMIC_OBSTACLE = False

# Initial state
init_state = np.array([[0., 0., 0.],
                       [1., 0., 0.],
                       [1., .5, 0.],
                       [0., .5, 0.]])  # px, py, theta

goal = np.array([[3.5, 2., 0.], [3.5, 1., 0.],[4., 1., 0.], [4., 2., 0.]]).flatten()  # px, py, theta

OBSTACLES = np.array([[2., 0., 0.], [2., 2.4, 0.]]).flatten()
OBSTACLE_NUM = int(OBSTACLES.shape[0] / 3)
OBSTACE_RANGE = 0.5

# goal = (init_state - np.average(init_state, axis=0)) @ np.array() + np.array([3.25, 1.25, 0.])

# Formation definition
ELL_EPS = 0.0
HALF_WIDTH = 1.0 + ELL_EPS
HALF_HEIGHT = 0.7 + ELL_EPS
ORIENTATION_OFFSET = 0.44
# NEIGHBOR_DIST = 0.7
nd_1 = 1.
nd_2 = 0.5
nd_3 = 1.118
EPSILON = 0.05

ADJACENCY = np.array([[0., nd_1, nd_3, nd_2],
                      [nd_1, 0., nd_2, 0.],
                      [nd_3, nd_2, 0., nd_1],
                      [nd_2, 0., nd_1, 0.]])

# Robot Models
MAX_VEL = 1.2
MAX_ROT = 5.
Rsi = 0.21 + OBSTACE_RANGE

# Gamma definition
b1 = 0.1
b2 = 1000
order = 3

# Define proportional gain
TIME_VARYING_GAIN = True
K = 0.8
BETA = 1.0

# Define Field size for plotting (should be in tuple)
field_x = (-1, 6)
field_y = (-1, 6)

# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state):
    """
    Compute the control input

    :param robot_state:
    :return:
    """
    ellipse = np.zeros(CLUSTER_NUM * 3)
    robot_per_cluster = int(ROBOT_NUM/CLUSTER_NUM)

    # Calculate Ellipse
    for i in range(CLUSTER_NUM):
        for j in range(robot_per_cluster):
            ellipse[3 * i: 3 * i + 2] += robot_state[3 * (robot_per_cluster * i + j):3 * (robot_per_cluster * i + j) + 2] / robot_per_cluster
        ellipse[3 * i + 2] = np.arctan2(ellipse[3 * i + 1] - robot_state[3 * robot_per_cluster * i + 1], ellipse[3 * i] - robot_state[3 * robot_per_cluster * i]) - ORIENTATION_OFFSET

    current_input = np.zeros(ROBOT_NUM * 3)
    # Proportional gain computation
    for i in range(ROBOT_NUM):
        distance = calculate_distance(robot_state[3 * i:3 * i + 2], desired_state[3 * i:3 * i + 2])
        if TIME_VARYING_GAIN:
            GAIN = K * (1 - np.exp(-BETA * distance)) / distance
        else:
            GAIN = K
        current_input[3 * i:3 * i + 3] = GAIN * (desired_state[3 * i:3 * i + 3] - robot_state[3 * i:3 * i + 3])

    # For final input and dynamic estimation
    current_input = saturate_velocity(current_input)
    temporary_input = current_input.copy()

    stacked_h = np.empty((0, 1))

    for i in range(CLUSTER_NUM):
        for j in range(robot_per_cluster):
            robot_index = robot_per_cluster * i + j
            h = np.empty((0, 1))
            H = np.empty((0, 2))
            temp_robot_state = robot_state[3 * robot_index:3 * robot_index + 2]

            # Distance to static obstacles
            for k in range(OBSTACLE_NUM):
                obstacle = OBSTACLES[3 * k:3 * k + 2]
                obs_distance = calculate_distance(temp_robot_state, obstacle)
                h = np.vstack((h, b2 * ((obs_distance ** 2 - Rsi ** 2) ** order)))
                H = np.vstack((H, -2 * (temp_robot_state - obstacle)))

            # Distance to other cluster
            for k in range(CLUSTER_NUM):
                if i == k:
                    continue

                A = formation_derivative(temp_robot_state, ellipse[3 * k: 3 * k + 3])
                # Estimated dynamic
                delta = np.zeros(2)
                for l in range(robot_per_cluster):
                    delta += temporary_input[3 * (robot_per_cluster * k + l):3 * (robot_per_cluster * k + l) + 2] / robot_per_cluster
                delta = A @ delta if DYNAMIC_OBSTACLE else 0.

                h = np.vstack((h, b1 * ((formation_distance(temp_robot_state, ellipse[3 * k: 3 * k + 3])) ** order) - delta))
                H = np.vstack((H, -A))

            # Distance in cluster
            for k in range(robot_per_cluster):
                # if j == k:
                #     continue

                neighbor_dist = ADJACENCY[j][k]

                if neighbor_dist == 0.:
                    continue

                neighbor_index = robot_per_cluster * i + k
                neighbor = robot_state[3 * neighbor_index:3 * neighbor_index + 2]
                difference_state = temp_robot_state - neighbor
                distance = np.linalg.norm(difference_state)
                delta = difference_state.T @ temporary_input[3 * neighbor_index:3 * neighbor_index + 2] if DYNAMIC_OBSTACLE else 0.
                # print(neighbor_dist, distance)
                h = np.vstack((h, b2 * ((distance ** 2 - (neighbor_dist - EPSILON) ** 2) ** order) - delta))
                h = np.vstack((h, b2 * ((-distance ** 2 + (neighbor_dist + EPSILON) ** 2) ** order) + delta))
                H = np.vstack((H, -2 * difference_state))
                H = np.vstack((H, 2 * difference_state))

            stacked_h = np.vstack((stacked_h, h))

            u_gtg = current_input[3 * robot_index:3 * robot_index + 2]

            Q_mat = 2 * cvxopt.matrix(np.eye(2), tc='d')
            c_mat = -2 * cvxopt.matrix(u_gtg, tc='d')

            H_mat = cvxopt.matrix(H, tc='d')
            b_mat = cvxopt.matrix(h, tc='d')

            # Find u*
            cvxopt.solvers.options['show_progress'] = False
            sol = cvxopt.solvers.qp(Q_mat, c_mat, H_mat, b_mat, verbose=False)

            current_input[3 * robot_index:3 * robot_index + 2] = np.array([sol['x'][0], sol['x'][1]])

    return saturate_velocity(current_input), ellipse, stacked_h[stacked_h != 0.]


def calculate_distance(former, latter):
    """
    Calculate Euclidean distance

    :param former:
    :param latter:
    :return:
    """
    distance = np.sqrt((former[0] - latter[0]) ** 2 + (former[1] - latter[1]) ** 2)
    return distance.item()


def formation_distance(former, latter):
    """
    Calculate Euclidean distance bet ween agent and cluster

    :param former:
    :param latter:
    :return:
    """
    A = np.array([[1/HALF_WIDTH, 0], [0, 1/HALF_HEIGHT]])
    rotation = np.array([[np.cos(-latter[2]), -np.sin(-latter[2])], [np.sin(-latter[2]), np.cos(-latter[2])]])
    result = np.linalg.norm(A @ rotation @ (former[0:2] - latter[0:2])) ** 2 - 1
    return result


def formation_derivative(former, latter):
    """
    Calculate the derivative of formation distance

    :param former:
    :param latter:
    :return:
    """
    theta = latter[2]
    # Good luck with derivative
    A = 2 * np.array([[(np.cos(theta)/HALF_WIDTH) ** 2 + (np.sin(theta)/HALF_HEIGHT              ) ** 2, np.cos(theta) * np.sin(theta) * (1/(HALF_WIDTH ** 2) - 1/(HALF_HEIGHT ** 2))],
                      [np.cos(theta) * np.sin(theta) * (1/(HALF_WIDTH ** 2) - 1/(HALF_HEIGHT ** 2)), (np.sin(theta)/HALF_WIDTH) ** 2 + (np.cos(theta)/HALF_HEIGHT) ** 2]])
    result = A @ (former[0:2] - latter[0:2])
    return result


def saturate_velocity(velocity):
    """
    Regulate the control input to comply with robot constraints

    :param velocity:
    :return:
    """
    for i in range(ROBOT_NUM):
        linear_velocity = np.sqrt(velocity[3 * i] ** 2 + velocity[3 * i + 1] ** 2)

        if linear_velocity > MAX_VEL:
            velocity[3 * i] /= linear_velocity / MAX_VEL
            velocity[3 * i + 1] /= linear_velocity / MAX_VEL

        velocity[3 * i + 2] = velocity[3 * i + 2] if abs(velocity[3 * i + 2]) <= MAX_ROT else np.sign(velocity[3 * i + 2]) * MAX_ROT

    return velocity


# MAIN SIMULATION COMPUTATION
# ---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max / Ts)  # Total Step for simulation

    # Initialize robots' state and goals (Single Integrator)
    robot_state = init_state.copy().flatten()  # numpy array for [px, py, theta]
    desired_state = goal.copy().flatten()

    robot_per_cluster = int(ROBOT_NUM / CLUSTER_NUM)
    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, 3 * ROBOT_NUM))  # for [vx, vy, omega] vs iteration time
    ell_history = np.zeros((sim_iter, 3 * CLUSTER_NUM))
    h_history = np.zeros((sim_iter, ROBOT_NUM * OBSTACLE_NUM + CLUSTER_NUM * ((CLUSTER_NUM - 1) * robot_per_cluster + np.count_nonzero(ADJACENCY) * 2)))

    ellipse = np.zeros(CLUSTER_NUM * 3)

    for i in range(CLUSTER_NUM):
        for j in range(robot_per_cluster):
            ellipse[3 * i: 3 * i + 2] += robot_state[3 * (robot_per_cluster * i + j):3 * (robot_per_cluster * i + j) + 2] / robot_per_cluster
        ellipse[3 * i + 2] = np.arctan2(ellipse[3 * i + 1] - robot_state[3 * robot_per_cluster * i + 1], ellipse[3 * i] - robot_state[3 * robot_per_cluster * i]) - ORIENTATION_OFFSET


    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_input = ['omnidirectional'] * ROBOT_NUM
        sim_visualizer = sim_mobile_robots(sim_input)  # Omnidirectional Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.show_goal(desired_state)
        sim_visualizer.show_formation(ellipse, (HALF_WIDTH - ELL_EPS) * 2, (HALF_HEIGHT - ELL_EPS) * 2)

        for i in range(OBSTACLE_NUM):
            sim_visualizer.ax.add_patch(plt.Circle(OBSTACLES[3 * i:3 * i + 3], OBSTACE_RANGE, color='r'))
            sim_visualizer.ax.add_patch(plt.Circle(OBSTACLES[3 * i:3 * i + 3], Rsi, color='r', fill=False))

    for it in range(sim_iter):
        current_time = it * Ts

        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # Compute control input
        current_input, ellipse, h = compute_control_input(desired_state, robot_state)

        # record the computed input at time-step t
        input_history[it] = current_input
        ell_history[it] = ellipse
        h_history[it] = h

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(current_time)
            sim_visualizer.update_goal(desired_state)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data
            sim_visualizer.update_formation(ellipse)

        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        robot_state = robot_state + Ts * current_input  # will be used in the next iteration
        for i in range(ROBOT_NUM):
            robot_state[i * 3 + 2] = ((robot_state[i * 3 + 2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history, ell_history, h_history


if __name__ == '__main__':
    # Call main computation for robot simulation
    state_history, goal_history, input_history, ell_history, h_history = simulate_control()
    # ADDITIONAL PLOTTING
    # ----------------------------------------------
    t = [i * Ts for i in range(round(t_max / Ts))]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    for i in range(ROBOT_NUM):
        ax.plot(t, input_history[:, i * 3], label=f'vx_{i + 1} [m/s]')
        ax.plot(t, input_history[:, 1 + i * 3], label=f'vy_{i + 1} [m/s]')
        # ax.plot(t, input_history[:, 2 + i * 3], label=f'omega{i + 1} [rad/s]')

    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.grid()

    # Plot historical data of state
    fig3 = plt.figure(3)
    ax = plt.gca()
    for i in range(ROBOT_NUM):
        ax.plot(t, state_history[:, i * 3], label=f'px_{i + 1} [m]')
        ax.plot(t, state_history[:, 1 + i * 3], label=f'py_{i + 1} [m]')
        ax.plot(t, goal_history[:, i * 3], '--', label=f'gx_{i + 1} [m/s]')
        ax.plot(t, goal_history[:, 1 + i * 3], '--', label=f'gy_{i + 1} [m/s]')

    ax.set(xlabel="t [s]", ylabel="state")
    plt.legend()
    plt.grid()

    fig4 = plt.figure(4)
    ax = plt.gca()
    for i in range(CLUSTER_NUM):
        ax.plot(t, ell_history[:, i * 3], label=f'x_{i + 1} [m]')
        ax.plot(t, ell_history[:, 1 + i * 3], label=f'y_{i + 1} [m]')
        ax.plot(t, ell_history[:, 2 + i * 3], label=f'angle_{i + 1} [rad]')

    ax.set(xlabel="t [s]", ylabel="state")
    plt.legend()
    plt.grid()

    # Plot historical data of h
    fig5 = plt.figure(5)
    ax = plt.gca()
    for i in range(ROBOT_NUM * (CLUSTER_NUM - 1 + int(ROBOT_NUM / CLUSTER_NUM))):
        ax.plot(t, h_history[:, i], label=f'h of Agent {int(i / int(CLUSTER_NUM - 1 + int(ROBOT_NUM / CLUSTER_NUM))) + 1}')

    ax.set(xlabel="t [s]", ylabel="h")
    plt.legend()
    plt.grid()

    plt.show()