# Fundamental of Mobile Robot - AUT.710 - 2023
# Hoang Pham, Nadun Ranasinghe, Dong Le

import cvxopt
import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robots import sim_mobile_robots

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 15 # total simulation duration in seconds
# Set initial state !!! DONT CHANGE THESE
RANDOM_INIT = False
SYMMETRIC_INIT = False
SYMMETRIC_GOAL = False

# Robot model
ROBOT_RADIUS = 0.21
WHEEL_RADIUS = 0.1
MAX_SPEED = 10.
b = 10
order = 3
# TOLERANCE = 0.01

# Alternative control point
l = ROBOT_RADIUS * 0.5
Rsi = ROBOT_RADIUS + l

# Formation definition
ELL_EPS = 0.00
HALF_WIDTH = 0.9 + ELL_EPS
HALF_HEIGHT = 0.6 + ELL_EPS
NEIGHBOR_DIST = 2.5 * Rsi
EPSILON = 0.05

# Initial state
init_state = np.array([[0., 0., 0.], [0., 0. + NEIGHBOR_DIST, 0.], \
                       [0., 5., 0.], [0., 5. - NEIGHBOR_DIST, 0.]]).flatten()  # px, py, theta

goal = np.array([[5., 4., 0.], [5. - NEIGHBOR_DIST, 4., 0.], \
                 [5., 0., 0.], [5. - NEIGHBOR_DIST, 0., 0.]]).flatten()  # px, py, theta

IS_SHOWING_2DVISUALIZATION = True
ROBOT_NUM = 4
CLUSTER_NUM = 2


# Define static K
TIME_VARYING_GAIN = True
K = 1.2
BETA = 1.0

# Define Field size for plotting (should be in tuple)
field_x = (-1, 6)
field_y = (-1, 6)

# Disturbances, no use
SALT = 0.

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
        ellipse[3 * i + 2] = np.arctan2(ellipse[3 * i + 1] - robot_state[3 * robot_per_cluster * i + 1], ellipse[3 * i] - robot_state[3 * robot_per_cluster * i])

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
    temporary_input = current_input.copy()

    stacked_h = np.empty((0, 1))

    for i in range(CLUSTER_NUM):
        for j in range(robot_per_cluster):
            robot_index = robot_per_cluster * i + j
            h = np.empty((0, 1))
            H = np.empty((0, 2))
            temp_robot_state = robot_state[3 * robot_index:3 * robot_index + 2]

            # Distance to other cluster
            for k in range(CLUSTER_NUM):
                if i == k:
                    continue

                A = formation_derivative(temp_robot_state, ellipse[3 * k: 3 * k + 3])
                # Estimated dynamic
                delta = np.zeros(2)
                for l in range(robot_per_cluster):
                    delta += temporary_input[3 * (robot_per_cluster * k + l):3 * (robot_per_cluster * k + l) + 2] / robot_per_cluster
                delta = A @ delta

                h = np.vstack((h, b * ((formation_distance(temp_robot_state, ellipse[3 * k: 3 * k + 3])) ** order) - delta))
                H = np.vstack((H, -A))

            # Distance in cluster
            for k in range(robot_per_cluster):
                if j == k:
                    continue

                neighbor_index = robot_per_cluster * i + k
                neighbor = robot_state[3 * neighbor_index:3 * neighbor_index + 2]
                difference_state = temp_robot_state - neighbor
                distance = np.linalg.norm(difference_state)
                delta = difference_state.T @ temporary_input[3 * neighbor_index:3 * neighbor_index + 2]

                h = np.vstack((h, b * ((distance ** 2 - (NEIGHBOR_DIST - EPSILON) ** 2) ** order) - delta))
                h = np.vstack((h, b * ((-distance ** 2 + (NEIGHBOR_DIST + EPSILON) ** 2) ** order) + delta))
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

    return current_input, ellipse, stacked_h[stacked_h != 0.]


def caster_transform(robot_state):
    uni_state = np.zeros(3 * ROBOT_NUM)
    for i in range(ROBOT_NUM):
        idx = 3 * i
        x_agent = robot_state[idx:(idx + 2)]
        theta = robot_state[idx + 2]
        S = x_agent + np.array([l * np.cos(theta), l * np.sin(theta)])

        uni_state[idx:(idx + 2)] = S
        uni_state[idx + 2] = robot_state[idx + 2]
    return uni_state


def control_transform(control_input, robot_state):
    current_input = np.zeros(8)
    for i in range(ROBOT_NUM):
        theta = robot_state[i * 3 + 2]
        current_input[2 * i:2 * i + 2] = np.array([[1, 0], [0, 1 / l]]) @ np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) @ control_input[3 * i:3 * i + 2]

    return current_input


def calculate_distance(former, latter):
    # Calculate Euclidean distance
    distance = np.sqrt((former[0] - latter[0]) ** 2 + (former[1] - latter[1]) ** 2)
    return distance.item()


def formation_distance(former, latter):
    # Calculate Euclidean distance
    A = np.array([[1/HALF_WIDTH, 0], [0, 1/HALF_HEIGHT]])
    rotation = np.array([[np.cos(-latter[2]), -np.sin(-latter[2])], [np.sin(-latter[2]), np.cos(-latter[2])]])
    result = np.linalg.norm(A @ rotation @ (former[0:2] - latter[0:2])) ** 2 - 1
    return result


def formation_derivative(former, latter):
    # Calculate Euclidean distance
    theta = latter[2]

    # Good luck with derivative
    A = 2 * np.array([[(np.cos(theta)/HALF_WIDTH) ** 2 + (np.sin(theta)/HALF_HEIGHT) ** 2, np.cos(theta) * np.sin(theta) * (1/(HALF_WIDTH ** 2) - 1/(HALF_HEIGHT ** 2))],
                      [np.cos(theta) * np.sin(theta) * (1/(HALF_WIDTH ** 2) - 1/(HALF_HEIGHT ** 2)), (np.sin(theta)/HALF_WIDTH) ** 2 + (np.cos(theta)/HALF_HEIGHT) ** 2]])
    result = A @ (former[0:2] - latter[0:2])
    return result

# No use
def random_initial_state():
    random_state = np.zeros(ROBOT_NUM * 3)
    range_x = field_x[1] - field_x[0]
    range_y = field_y[1] - field_y[0]
    range_theta = 2 * np.pi

    for i in range(ROBOT_NUM):
        random_state[3 * i] = round(np.random.rand() * range_x + field_x[0], 4)
        random_state[3 * i + 1] = round(np.random.rand() * range_y + field_y[0], 4)
        random_state[3 * i + 2] = round(np.random.rand() * range_theta - np.pi, 4)

    return random_state

# No use
def symmetric_initial_state():
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

    return random_state


def symmetric_goal(robot_state):
    goal_state = np.zeros(ROBOT_NUM * 3)
    range_x = field_x[1] + field_x[0]
    range_y = field_y[1] + field_y[0]

    for i in range(ROBOT_NUM):
        goal_state[3 * i] = range_x - robot_state[3 * i]
        goal_state[3 * i + 1] = range_y - robot_state[3 * i + 1]
        goal_state[3 * i + 2] = -robot_state[3 * i + 2]

    return goal_state

def saturate(velocity):
    # Calculate the wheel velocity
    wheel_speed = np.zeros(np.shape(velocity))
    for i in range(ROBOT_NUM):
        wr = (2 * velocity[2 * i] + velocity[2 * i + 1] * ROBOT_RADIUS) / (2 * WHEEL_RADIUS)
        wl = (2 * velocity[2 * i] - velocity[2 * i + 1] * ROBOT_RADIUS) / (2 * WHEEL_RADIUS)

        # Saturate if greater than threshold
        wr = wr if abs(wr) < MAX_SPEED else np.sign(wr) * MAX_SPEED
        wl = wl if abs(wl) < MAX_SPEED else np.sign(wl) * MAX_SPEED

        # Recalculate the control input
        velocity[2 * i + 1] = (wr - wl) * WHEEL_RADIUS / ROBOT_RADIUS
        velocity[2 * i + 1] = (wr + wl) * WHEEL_RADIUS / 2
        wheel_speed[2 * i:2 * i + 2] = np.array([wl, wr])

    return velocity, wheel_speed

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
        # desired_state = random_initial_state()
        desired_state = goal.copy()

    robot_per_cluster = int(ROBOT_NUM / CLUSTER_NUM)
    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, 2 * ROBOT_NUM))  # for [vx, vy, omega] vs iteration time
    ell_history = np.zeros((sim_iter, 3 * CLUSTER_NUM))
    h_history = np.zeros((sim_iter, ROBOT_NUM * (CLUSTER_NUM - 1 + 2 * (robot_per_cluster - 1))))
    wheel_history = np.zeros((sim_iter, 2 * ROBOT_NUM))

    ellipse = np.zeros(CLUSTER_NUM * 3)

    for i in range(CLUSTER_NUM):
        for j in range(robot_per_cluster):
            ellipse[3 * i: 3 * i + 2] += robot_state[3 * (robot_per_cluster * i + j):3 * (robot_per_cluster * i + j) + 2] / robot_per_cluster
        # ellipse[3 * i + 2] = np.arctan(ellipse[3 * i + 1] / ellipse[3 * i]) + np.pi/2
        ellipse[3 * i + 2] = np.arctan2(ellipse[3 * i + 1] - robot_state[3 * robot_per_cluster * i + 1], ellipse[3 * i] - robot_state[3 * robot_per_cluster * i])


    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_input = ['unicycle'] * ROBOT_NUM
        sim_visualizer = sim_mobile_robots(sim_input)  # Omnidirectional Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.show_goal(desired_state)
        sim_visualizer.show_formation(ellipse, (HALF_WIDTH - ELL_EPS) * 2, (HALF_HEIGHT - ELL_EPS) * 2)

    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state
        uni_state = caster_transform(robot_state)

        # Compute control input
        current_input, ellipse, h = compute_control_input(desired_state, uni_state)

        # record the computed input at time-step t
        current_input = control_transform(current_input, robot_state)
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
        # robot_state = robot_state + Ts * current_input  # will be used in the next iteration

        for i in range(ROBOT_NUM):
            B = np.array([[np.cos(robot_state[3 * i + 2]), 0], [np.sin(robot_state[3 * i + 2]), 0], [0, 1]])
            # print(B @ current_input[2 * i:2 * i + 2])
            robot_state[3 * i:3 * i + 3] = robot_state[3 * i:3 * i + 3] + Ts * (B @ current_input[2 * i:2 * i + 2])  # will be used in the next iteration
            robot_state[3 * i + 2] = ((robot_state[i * 3 + 2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

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
        ax.plot(t, input_history[:, i * 2], label=f'vx_{i + 1} [m/s]')
        ax.plot(t, input_history[:, 1 + i * 2], label=f'vy_{i + 1} [m/s]')
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
        if i % 3 == 2:
            continue
        ax.plot(t, h_history[:, i], label=f'h of Agent {int(i / int(CLUSTER_NUM - 1 + int(ROBOT_NUM / CLUSTER_NUM))) + 1}')

    ax.set(xlabel="t [s]", ylabel="h")
    plt.legend()
    plt.grid()

    plt.show()