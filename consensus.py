# Fundamental of Mobile Robot - AUT.710 - 2023
# Hoang Pham, Nadun Ranasinghe, Dong Le

import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robots import sim_mobile_robots

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 5 # total simulation duration in seconds
# Set initial state
init_state = np.array([[-1., 4., 0.], [4., 0., 0.], [-1., 2., 0.], [4., 4., 0.]]).flatten()  # px, py, theta

IS_SHOWING_2DVISUALIZATION = True
ROBOT_NUM = 4
LAPLACIAN_MAT = np.array([[2., -1., 0., -1.],
                          [-1., 2., -1., 0.],
                          [0., -1., 2., -1.],
                          [-1., 0., -1., 2.]])
B_MAT = np.array([2., -2., 0., -2., -2., 0., -2., 2., 0., 2., 2., 0.])
GAMMA = 0.2

# Define Field size for plotting (should be in tuple)
field_x = (-2, 5)
field_y = (-2, 5)


# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_control_input(robot_state):
    # Check if using static gain
    # print(robot_state)

    P_mat_1 = np.zeros((ROBOT_NUM, ROBOT_NUM))
    P_mat_2 = np.eye(ROBOT_NUM, ROBOT_NUM)
    P_mat_3 = -LAPLACIAN_MAT
    P_mat_4 = -GAMMA * LAPLACIAN_MAT
    P_mat = np.block([[P_mat_1, P_mat_2], [P_mat_3, P_mat_4]])
    B_mat = np.block([np.zeros(ROBOT_NUM * 3), B_MAT])
    # print(np.kron(P_mat, np.eye(3, 3)))
    new_state = np.kron(P_mat, np.eye(3, 3)) @ robot_state.flatten() + B_mat
    current_input = new_state[3 * ROBOT_NUM:]
    # print(current_input)
    return current_input


def calculate_distance(former, latter):
    # Calculate Euclidean distance
    distance = np.sqrt((former[0] - latter[0]) ** 2 + (former[1] - latter[1]) ** 2)
    return distance.item()


# MAIN SIMULATION COMPUTATION
# ---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max / Ts)  # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy()  # numpy array for [px, py, theta]
    # desired_state = np.array([-2., 1., 0.])  # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    input_history = np.zeros((sim_iter, 3 * ROBOT_NUM))  # for [vx, vy, omega] vs iteration time

    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_visualizer = sim_mobile_robots(['omnidirectional', 'omnidirectional', 'omnidirectional', 'omnidirectional'])  # Omnidirectional Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area

    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state

        # Compute control input
        current_input = compute_control_input(np.concatenate((robot_state, input_history[it-1])))

        # record the computed input at time-step t
        input_history[it] = current_input

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(current_time)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data

        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        # for i in range(ROBOT_NUM):
        robot_state = robot_state + Ts * current_input  # will be used in the next iteration
        for i in range(ROBOT_NUM):
            robot_state[i * 3 + 2] = ((robot_state[i * 3 + 2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, input_history


if __name__ == '__main__':
    # Call main computation for robot simulation
    state_history, input_history = simulate_control()
    # ADDITIONAL PLOTTING
    # ----------------------------------------------
    t = [i * Ts for i in range(round(t_max / Ts))]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    for i in range(ROBOT_NUM):
        ax.plot(t, input_history[:, i * 3], label=f'vx{i + 1} [m/s]')
        ax.plot(t, input_history[:, 1 + i * 3], label=f'vy{i + 1} [m/s]')
        ax.plot(t, input_history[:, 2 + i * 3], label=f'omega{i + 1} [rad/s]')

    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.grid()

    # Plot historical data of state
    fig3 = plt.figure(3)
    ax = plt.gca()
    for i in range(ROBOT_NUM):
        ax.plot(t, state_history[:, i * 3], label=f'px{i + 1} [m]')
        ax.plot(t, state_history[:, 1 + i * 3], label=f'py{i + 1} [m]')
        ax.plot(t, state_history[:, 2 + i * 3], label=f'theta{i + 1} [rad]')

    ax.set(xlabel="t [s]", ylabel="state")
    plt.legend()
    plt.grid()

    plt.show()