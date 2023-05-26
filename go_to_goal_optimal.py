"""
Fundamental of Mobile Robot - AUT.710 - 2023
Mini Project - Multiple robots go to goal with dynamic QP
Hoang Pham, Nadun Ranasinghe, Dong Le
"""

import cvxopt
import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robots import sim_mobile_robots

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 10 # total simulation duration in seconds
IS_SHOWING_2DVISUALIZATION = True
ROBOT_NUM = 2
RANDOM_INIT = False
SYMMETRIC_INIT = True
SYMMETRIC_GOAL = True
DYNAMIC_OBS = True

# Set initial state
init_state = np.array([[-1., 4., 0.], [4., 0., 0.], [-1., 2., 0.],
                       [4., 4., 0.], [3., 1., 0.], [4., -1., 0.]]).flatten()  # px, py, theta

# Robot Models
MAX_VEL = 1.5
MAX_ROT = 5.
Rsi = 0.5
TOLERANCE = 0.01

# Gamma definition
b = 10
order = 3

# Define proportional gain
TIME_VARYING_GAIN = True
K = 1.2
BETA = 1.0

# Define Field size for plotting (should be in tuple)
field_x = (-2, 5)
field_y = (-2, 5)

# Disturbances
SALT = 0.

# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state):
    """
    Compute the control input

    :param desired_state:
    :param robot_state:
    :return:
    """
    for i in range(ROBOT_NUM):
        robot_state[3 * i] += np.random.rand() * SALT - SALT / 2
        robot_state[3 * i + 1] += np.random.rand() * SALT - SALT / 2

    current_input = np.zeros(ROBOT_NUM * 3)
    # Proportional gain computation
    for i in range(ROBOT_NUM):
        distance = calculate_distance(robot_state[3 * i:3 * i + 2], desired_state[3 * i:3 * i + 2])
        if TIME_VARYING_GAIN:
            GAIN = K * (1 - np.exp(-BETA * distance)) / distance
        else:
            GAIN = K
        current_input[3 * i:3 * i + 3] = GAIN * (desired_state[3 * i:3 * i + 3] - robot_state[3 * i:3 * i + 3])

    current_input = saturate_velocity(current_input)
    temporary_input = current_input.copy()

    stacked_h = np.empty((0))

    for i in range(ROBOT_NUM):
        d = np.zeros(ROBOT_NUM)
        h = np.zeros(ROBOT_NUM)
        H = np.zeros((ROBOT_NUM, 2))

        for j in range(ROBOT_NUM):
            if i == j:
                continue

            obs_distance = calculate_distance(robot_state[3 * i: 3 * i + 2], robot_state[3 * j: 3 * j + 2])

            H[j, :] = - 2 * (robot_state[3 * i: 3 * i + 2] - robot_state[3 * j: 3 * j + 2])
            d[j] = 2 * (robot_state[3 * i: 3 * i + 2] - robot_state[3 * j: 3 * j + 2]).T @ temporary_input[3 * j:3 * j + 2]

            if DYNAMIC_OBS:
                h[j] = b * ((obs_distance ** 2 - (Rsi + j * 0.05) ** 2) ** order) - d[j]
            else:
                h[j] = b * ((obs_distance ** 2 - Rsi ** 2) ** order)

        stacked_h = np.append(stacked_h, h, axis=0)

        # Regulated u_gtg
        u_gtg = temporary_input[3 * i: 3 * i + 2]

        # Construct Q, H, b, c for QP-based controller
        Q_mat = 2 * cvxopt.matrix(np.eye(2), tc='d')
        c_mat = -2 * cvxopt.matrix(u_gtg, tc='d')

        H_mat = cvxopt.matrix(H, tc='d')
        b_mat = cvxopt.matrix(h, tc='d')

        # Find u*
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(Q_mat, c_mat, H_mat, b_mat, verbose=False)
        current_input[3 * i: 3 * i + 2] = np.array([sol['x'][0], sol['x'][1]])

    return saturate_velocity(current_input), stacked_h[stacked_h != 0.]


def calculate_distance(former, latter):
    """
    Calculate Euclidean distance

    :param former:
    :param latter:
    :return:
    """
    distance = np.sqrt((former[0] - latter[0]) ** 2 + (former[1] - latter[1]) ** 2)
    return distance.item()


def random_initial_state():
    """
    Generate random position

    :return:
    """
    random_state = np.zeros(ROBOT_NUM * 3)
    range_x = field_x[1] - field_x[0]
    range_y = field_y[1] - field_y[0]
    range_theta = 2 * np.pi

    for i in range(ROBOT_NUM):
        random_state[3 * i] = round(np.random.rand() * range_x + field_x[0], 4)
        random_state[3 * i + 1] = round(np.random.rand() * range_y + field_y[0], 4)
        random_state[3 * i + 2] = round(np.random.rand() * range_theta - np.pi, 4)

    return random_state


def symmetric_initial_state():
    """
    Generate symmetric initial position

    :param velocity:
    :return:
    """
    random_state = np.zeros(ROBOT_NUM * 3)
    range_x = field_x[1] - field_x[0]
    range_y = field_y[1] - field_y[0]
    center = 0.5 * np.array([field_x[1] + field_x[0], field_y[1] + field_y[0]])
    range_theta = 2 * np.pi
    radius = min(range_x, range_y) * 0.4
    angle = [2 * i * np.pi / ROBOT_NUM for i in range(ROBOT_NUM)]

    for i in range(ROBOT_NUM):
        random_state[3 * i] = round(center[0] + radius * np.cos(angle[i]), 4)
        random_state[3 * i + 1] = round(center[1] + radius * np.sin(angle[i]), 4)
        random_state[3 * i + 2] = round(np.random.rand() * range_theta - np.pi, 4)

    print("Initial state:\n", random_state, "\n")
    return random_state


def symmetric_goal(robot_state):
    """
    Generate symmetric goal

    :param velocity:
    :return:
    """
    goal_state = np.zeros(ROBOT_NUM * 3)
    range_x = field_x[1] + field_x[0]
    range_y = field_y[1] + field_y[0]

    for i in range(ROBOT_NUM):
        goal_state[3 * i] = range_x - robot_state[3 * i]
        goal_state[3 * i + 1] = range_y - robot_state[3 * i + 1]
        goal_state[3 * i + 2] = -robot_state[3 * i + 2]

    print("Goal state:\n", goal_state, "\n")
    return goal_state


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

    # Initialize robot's state (Single Integrator)
    if RANDOM_INIT:
        robot_state = random_initial_state()
    elif SYMMETRIC_INIT:
        robot_state = symmetric_initial_state()
    else:
        robot_state = init_state.copy()  # numpy array for [px, py, theta]

    if SYMMETRIC_GOAL:
        desired_state = symmetric_goal(robot_state)
    else:
        desired_state = random_initial_state()

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, 3 * ROBOT_NUM))  # for [vx, vy, omega] vs iteration time
    h_history = np.zeros((sim_iter, ROBOT_NUM * (ROBOT_NUM - 1)))

    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_input = ['omnidirectional'] * ROBOT_NUM
        sim_visualizer = sim_mobile_robots(sim_input)  # Omnidirectional Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.show_goal(desired_state)

    for it in range(sim_iter):
        current_time = it * Ts

        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # Compute control input
        current_input, h = compute_control_input(desired_state, robot_state)

        # record the computed input at time-step t
        input_history[it] = current_input
        h_history[it] = h

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(current_time)
            sim_visualizer.update_goal(desired_state)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data

        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        robot_state = robot_state + Ts * current_input  # will be used in the next iteration
        for i in range(ROBOT_NUM):
            robot_state[i * 3 + 2] = ((robot_state[i * 3 + 2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, input_history, h_history


if __name__ == '__main__':
    # Call main computation for robot simulation
    state_history, input_history, h_history = simulate_control()
    # ADDITIONAL PLOTTING
    # ----------------------------------------------
    t = [i * Ts for i in range(round(t_max / Ts))]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    for i in range(ROBOT_NUM):
        ax.plot(t, input_history[:, i * 3], label=f'vx{i + 1} [m/s]')
        ax.plot(t, input_history[:, 1 + i * 3], label=f'vy{i + 1} [m/s]')
        # ax.plot(t, input_history[:, 2 + i * 3], label=f'omega{i + 1} [rad/s]')

    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.grid()

    # Plot historical data of state
    fig3 = plt.figure(3)
    ax = plt.gca()
    for i in range(ROBOT_NUM):
        ax.plot(t, state_history[:, i * 3], label=f'px{i + 1} [m]')
        ax.plot(t, state_history[:, 1 + i * 3], label=f'py{i + 1} [m]')
        # ax.plot(t, state_history[:, 2 + i * 3], label=f'theta{i + 1} [rad]')

    ax.set(xlabel="t [s]", ylabel="state")
    plt.legend()
    plt.grid()

    # Plot historical data of h
    fig4 = plt.figure(4)
    ax = plt.gca()
    for i in range(ROBOT_NUM * (ROBOT_NUM - 1)):
        ax.plot(t, h_history[:, i], label=f'h of Agent {int(i / (ROBOT_NUM - 1)) + 1}')

    ax.set(xlabel="t [s]", ylabel="h")
    plt.legend()
    plt.grid()

    plt.show()