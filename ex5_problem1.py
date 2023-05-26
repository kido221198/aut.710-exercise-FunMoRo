# Fundamental of Mobile Robot - AUT.710 - 2023
# Exercise 05 - Problem 01
# Hoang Pham, Nadun Ranasinghe

import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 15  # total simulation duration in seconds
# Set initial state
init_state = np.array([-2., 0.5, 0.])  # px, py, theta
obstacle = np.array([0., 0., 0.])
IS_SHOWING_2DVISUALIZATION = True

# Robot and environment constraints
RADIUS = 0.21
MAX_VEL = 0.5
MAX_ROT = 5
SAFE_DISTANCE = 0.8
EPSILON = 0.01

# Proportional gain for both controllers
GAIN = 5

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)


# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state, policy):
    obs_distance = calculate_distance(obstacle, robot_state)

    # Policy switching
    if policy == 'gtg' and obs_distance < SAFE_DISTANCE:
        policy = 'avo'
    elif policy == 'avo' and obs_distance > SAFE_DISTANCE + EPSILON:
        policy = 'gtg'

    # Compute the control input
    if policy == 'gtg':
        current_input = GAIN * (desired_state - robot_state)
    else:
        current_input = GAIN * (robot_state - obstacle)

    return saturate_velocity(current_input), policy, obs_distance


def calculate_distance(former, latter):
    # Calculate Euclidean distance
    distance = np.sqrt((former[0] - latter[0]) ** 2 + (former[1] - latter[1]) ** 2)
    return distance.item()


def saturate_velocity(velocity):
    # Regulate the control input to comply with robot constraints
    linear_velocity = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

    if linear_velocity > MAX_VEL:
        velocity[0] /= linear_velocity / MAX_VEL
        velocity[1] /= linear_velocity / MAX_VEL

    velocity[2] = velocity[2] if abs(velocity[2]) <= MAX_ROT else np.sign(velocity[2]) * MAX_ROT

    return velocity


# MAIN SIMULATION COMPUTATION
# ---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max / Ts)  # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy()  # numpy array for [px, py, theta]
    desired_state = np.array([2., -1., 0.])  # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, 3))  # for [vx, vy, omega] vs iteration time
    obs_history = np.zeros(sim_iter)
    policy = 'gtg'  # policy initialization

    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_visualizer = sim_mobile_robot('omnidirectional')  # Omnidirectional Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.ax.add_patch(plt.Circle((0, 0), 0.5, color='r'))
        sim_visualizer.ax.add_patch(plt.Circle((0, 0), SAFE_DISTANCE, color='r', fill=False))
        sim_visualizer.ax.add_patch(plt.Circle((0, 0), SAFE_DISTANCE + EPSILON, color='g', fill=False))
        sim_visualizer.show_goal(desired_state)

    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # COMPUTE CONTROL INPUT
        # ------------------------------------------------------------
        current_input, policy, obs_distance = compute_control_input(desired_state, robot_state, policy)
        # ------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input
        obs_history[it] = obs_distance

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(current_time)
            sim_visualizer.update_goal(desired_state)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data

        # --------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        robot_state = robot_state + Ts * current_input  # will be used in the next iteration
        robot_state[2] = ((robot_state[2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history, obs_history


if __name__ == '__main__':
    # Call main computation for robot simulation
    state_history, goal_history, input_history, obs_history = simulate_control()

    # ADDITIONAL PLOTTING
    # ----------------------------------------------
    t = [i * Ts for i in range(round(t_max / Ts))]

    # Plot historical data of control input
    fig2, ax = plt.subplots(2, 1)
    # ax = plt.gca()
    ax[0].plot(t, input_history[:, 0], label='vx [m/s]')
    ax[0].plot(t, input_history[:, 1], label='vy [m/s]')
    ax[1].plot(t, np.sqrt(input_history[:, 0] ** 2 + input_history[:, 1] ** 2), label='v [m/s]')
    ax[0].plot(t, input_history[:, 2], label='omega [rad/s]')
    ax[0].set(ylabel="control input")
    ax[1].set(xlabel="t [s]", ylabel="control input")
    fig2.legend()
    ax[0].grid()
    ax[1].grid()

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

    # Plot historical error
    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, goal_history[:, 0] - state_history[:, 0], label='error in x [m]')
    ax.plot(t, goal_history[:, 1] - state_history[:, 1], label='error in y [m]')
    ax.plot(t, obs_history, label='distance to obstacle [m]')
    ax.set(xlabel="t [s]", ylabel="distance")
    plt.legend()
    plt.grid()

    plt.show()
