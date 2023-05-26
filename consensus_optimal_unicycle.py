# Fundamental of Mobile Robot - AUT.710 - 2023
# Mini Project - Consensus with QP
# Hoang Pham, Nadun Ranasinghe, Dong Le

import cvxopt
import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robots import sim_mobile_robots

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 10 # total simulation duration in seconds

# Set initial state
RANDOM_INIT = True
init_state = np.array([[-1., 4., 0.], [4., 0., 0.], [-1., 2., 0.],
                       [4., 4., 0.], [3., 1., 0.], [4., -1., 0.]]).flatten()  # px, py, theta

IS_SHOWING_2DVISUALIZATION = True

ROBOT_NUM = 4
# LAPLACIAN_MAT = np.array([[2., -1., 0., 0., 0., -1.],
#                           [-1., 2., -1., 0., 0., 0.],
#                           [0., -1., 2., -1., 0., 0.],
#                           [0., 0., -1., 2., -1., 0.],
#                           [0., 0., 0., -1., 2., -1.],
#                           [-1., 0., 0., 0., -1., 2.]])

LAPLACIAN_MAT = np.array([[2., -1., 0., -1.],
                          [-1., 2., -1., 0.],
                          [0., -1., 2., -1.],
                          [-1., 0., -1., 2.]])

# B_MAT = np.array([-0.5, 0.87, 0.,
#                   0.5, 0.87, 0.,
#                   1., 0., 0.,
#                   0.5, -0.87, 0.,
#                   -0.5, -0.87, 0.,
#                   -1., 0., 0.]) * 2

B_MAT = np.array([-1., -1., 0., -1., 1., 0., 1., 1., 0., 1., -1., 0.])
GAMMA = 0.2
MAX_VEL = 2.
MAX_ROT = 5
Rsi = 0.5
b = 10
order = 3

# spec of unicycle
ROBOT_RADIUS = 0.08
l = 0.8 * ROBOT_RADIUS
WHEEL_RADIUS = 0.066/2

# Define controller parameters
K = 0.9
nu = 0.2
alpha = 7.

# Define static K
TIME_VARYING_GAIN = True
K = 0.9
BETA = 1.0

# Define Field size for plotting (should be in tuple)
field_x = (-2, 5)
field_y = (-2, 5)


# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_control_input(robot_state):
    # Check if using static gain
    # print(robot_state)

    # P_mat_1 = np.zeros((ROBOT_NUM, ROBOT_NUM))
    # P_mat_2 = np.eye(ROBOT_NUM, ROBOT_NUM)
    # P_mat_3 = -LAPLACIAN_MAT
    # P_mat_4 = -GAMMA * LAPLACIAN_MAT
    # P_mat = np.block([[P_mat_1, P_mat_2], [P_mat_3, P_mat_4]])
    # B_mat = np.block([np.zeros(ROBOT_NUM * 3), B_MAT])
    #
    # new_state = np.kron(P_mat, np.eye(3, 3)) @ robot_state.flatten() + B_mat

    new_state = -np.kron(LAPLACIAN_MAT, np.eye(3, 3)) @ robot_state.flatten() + B_MAT

    for i in range(ROBOT_NUM * 2):
        if TIME_VARYING_GAIN:
            GAIN = K * (1 - np.exp(-BETA * new_state[i])) / new_state[i]
        else:
            GAIN = K
        new_state[i] = new_state[i] * GAIN

    current_input = saturate_velocity(new_state)
    temporary_input = current_input.copy()

    stacked_h = np.empty((0))

    for i in range(ROBOT_NUM):
        h = np.zeros(ROBOT_NUM)
        d = np.zeros(ROBOT_NUM)
        H = np.zeros((ROBOT_NUM, 2))
        obs_distances = np.zeros(ROBOT_NUM)

        for j in range(ROBOT_NUM):
            if i == j:
                continue

            obs_distance = calculate_distance(robot_state[3 * i: 3 * i + 2], robot_state[3 * j: 3 * j + 2])

            H[j, :] = -2 * (robot_state[3 * i: 3 * i + 2] - robot_state[3 * j: 3 * j + 2])
            d[j] = 2 * (robot_state[3 * i: 3 * i + 2] - robot_state[3 * j: 3 * j + 2]) @ temporary_input[3 * j: 3 * j + 2]
            h[j] = b * ((obs_distance ** 2 - Rsi ** 2) ** order) - d[j]
            obs_distances[j] = obs_distance

        # Regulated u_gtg
        u_gtg = current_input[3 * i: 3 * i + 2]

        stacked_h = np.append(stacked_h, h, axis=0)

        # Construct Q, H, b, c for QP-based controller
        Q_mat = 2 * cvxopt.matrix(np.eye(2), tc='d')
        c_mat = -2 * cvxopt.matrix(u_gtg, tc='d')

        H_mat = cvxopt.matrix(H, tc='d')
        b_mat = cvxopt.matrix(h, tc='d')

        # Find u*
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(Q_mat, c_mat, H_mat, b_mat, verbose=False)

        current_input[3 * i: 3 * i + 2] = np.array([sol['x'][0], sol['x'][1]])

    # print(temporary_input - current_input)
    return saturate_velocity(current_input), stacked_h[stacked_h != 0.]


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


def saturate_velocity(velocity):
    # Regulate the control input to comply with robot constraints
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
    else:
        robot_state = init_state.copy()  # numpy array for [px, py, theta]
    # desired_state = np.array([-2., 1., 0.])  # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    input_history = np.zeros((sim_iter, 2 * ROBOT_NUM))  # for [vx, vy, omega] vs iteration time
    h_history = np.zeros((sim_iter, ROBOT_NUM * (ROBOT_NUM - 1)))

    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_input = ['unicycle'] * ROBOT_NUM
        sim_visualizer = sim_mobile_robots(sim_input)  # Omnidirectional Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area

    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state
        uni_state = caster_transform(robot_state)

        # Compute control input
        current_input, h = compute_control_input(uni_state)

        # record the computed input at time-step t
        current_input = control_transform(current_input, robot_state)
        input_history[it] = current_input
        h_history[it] = h

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(current_time)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data

        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        # for i in range(ROBOT_NUM):
        for i in range(ROBOT_NUM):
            index = 3 * i
            theta = robot_state[index + 2]
            angle_ = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
            robot_state[index:(index + 3)] = robot_state[index:(index + 3)] + Ts * (angle_ @ current_input[2 * i:2 * i + 2])  # will be used in the next iteration
            robot_state[index + 2] = ((robot_state[index + 2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

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
        ax.plot(t, input_history[:, i * 2], label=f'vx{i + 1} [m/s]')
        ax.plot(t, input_history[:, i * 2 + 1], label=f'vy{i + 1} [m/s]')
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