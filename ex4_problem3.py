# Fundamental of Mobile Robot - AUT.710 - 2023
# Exercise 04 - Problem 03
# Hoang Pham, Nadun Ranasinghe, Long Phan

import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = np.pi  # total simulation duration in seconds
# Set initial state
init_state = np.array([0., 0., -np.pi / 2])  # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Define static K
Kw = 2.29

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)


# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state):
    # initial numpy array for [vlin, omega]
    current_input = np.array([0., 0.])
    # Compute the control input
    angle = steering_angle(desired_state, robot_state) - robot_state[2]
    distance = calculate_distance(desired_state, robot_state)

    if angle > np.pi:
        angle = angle - 2 * np.pi
    elif angle < -np.pi:
        angle = angle + 2 * np.pi

    current_input[0] = 1. if distance > 0.05 else 0.
    current_input[1] = Kw * angle

    return current_input


def calculate_distance(former, latter):
    # Calculate Euclidean distance
    distance = np.sqrt((former[0] - latter[0]) ** 2 + (former[1] - latter[1]) ** 2)
    return distance.item()


def steering_angle(goal, pose):
    angle = np.arctan2(goal[1] - pose[1], goal[0] - pose[0])
    return angle


# MAIN SIMULATION COMPUTATION
# ---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max / Ts)  # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy()  # numpy array for [px, py, theta]
    desired_state = np.array([-1., 1., 1.])  # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, 2))  # for [vlin, omega] vs iteration time

    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_visualizer = sim_mobile_robot('unicycle')  # Unicycle Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.show_goal(desired_state)

    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # COMPUTE CONTROL INPUT
        # ------------------------------------------------------------
        current_input = compute_control_input(desired_state, robot_state)
        # ------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(current_time)
            sim_visualizer.update_goal(desired_state)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data

        # --------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        theta = robot_state[2]
        B = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
        robot_state = robot_state + Ts * (B @ current_input)  # will be used in the next iteration
        robot_state[2] = ((robot_state[2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

        # Update desired state if we consider moving goal position
        # desired_state = desired_state + Ts*(-1)*np.ones(len(robot_state))

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history


if __name__ == '__main__':
    # Call main computation for robot simulation
    state_history, goal_history, input_history = simulate_control()

    # ADDITIONAL PLOTTING
    # ----------------------------------------------
    t = [i * Ts for i in range(round(t_max / Ts))]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, input_history[:, 0], label='vx [m/s]')
    ax.plot(t, input_history[:, 1], label='omega [rad/s]')
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

    fig4, ax = plt.subplots(2, 1)
    # ax = plt.gca()
    ax[0].plot(t, goal_history[:, 0] - state_history[:, 0], label='error in x [m]')
    ax[1].plot(t, goal_history[:, 1] - state_history[:, 1], label='error in y [m]')
    ax[0].set(ylabel="error")
    ax[1].set(ylabel="error")
    fig4.legend()
    ax[0].grid()
    ax[1].grid()

    plt.show()
