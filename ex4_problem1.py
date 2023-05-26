# Fundamental of Mobile Robot - AUT.710 - 2023
# Exercise 04 - Problem 01
# Hoang Pham, Nadun Ranasinghe, Long Phan

import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = np.pi  # total simulation duration in seconds
# Set initial state
init_state = np.array([0., 0., 0.])  # px, py, theta
IS_SHOWING_2DVISUALIZATION = True
STATIC_GAIN = False

# Define static K
K = 1.73

# Define parameters for dynamic K
V0 = 20
BETA = 1

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)


# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state, current_time, static):
    # Check if using static gain
    if static:
        # Compute the control input
        current_input = K * (desired_state - robot_state)
    else:
        # Compute the gain
        e = calculate_distance(desired_state, robot_state)
        k = V0 * (1 - np.exp(-BETA * e)) / e

        # Compute the control input
        current_input = k * (desired_state - robot_state)

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
    desired_state = np.array([-2., 1., 0.])  # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, 3))  # for [vx, vy, omega] vs iteration time

    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_visualizer = sim_mobile_robot('omnidirectional')  # Omnidirectional Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.show_goal(desired_state)

    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # Compute control input
        current_input = compute_control_input(desired_state, robot_state, current_time, STATIC_GAIN)

        # record the computed input at time-step t
        input_history[it] = current_input

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(current_time)
            sim_visualizer.update_goal(desired_state)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data

        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        robot_state = robot_state + Ts * current_input  # will be used in the next iteration
        robot_state[2] = ((robot_state[2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history


if __name__ == '__main__':
    # Call main computation for robot simulation
    state_history, goal_history, input_history = simulate_control()
    print(goal_history[-1] - state_history[-1])
    # ADDITIONAL PLOTTING
    # ----------------------------------------------
    t = [i * Ts for i in range(round(t_max / Ts))]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, input_history[:, 0], label='vx [m/s]')
    ax.plot(t, input_history[:, 1], label='vy [m/s]')
    ax.plot(t, input_history[:, 2], label='omega [rad/s]')
    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.grid()

    # Plot historical data of state
    fig3 = plt.figure(3)
    ax = plt.gca()
    ax.plot(t, state_history[:, 0], label='px [m]')
    ax.plot(t, state_history[:, 1], label='py [m]')
    ax.plot(t, state_history[:, 2], label='theta [rad]')
    ax.plot(t, goal_history[:, 0], ':', label='goal px [m]')
    ax.plot(t, goal_history[:, 1], ':', label='goal py [m]')
    ax.plot(t, goal_history[:, 2], ':', label='goal theta [rad]')
    ax.set(xlabel="t [s]", ylabel="state")
    plt.legend()
    plt.grid()

    plt.show()