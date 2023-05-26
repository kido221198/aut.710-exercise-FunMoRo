# Fundamental of Mobile Robot - AUT.710 - 2023
# Exercise 05 - Problem 03
# Hoang Pham, Nadun Ranasinghe

import cvxopt
import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 30  # total simulation duration in seconds
# Set initial state
init_state = np.array([-2., -1, 0.])  # px, py, theta
obstacles = np.array([[-0.85, 0.3, 1.05], [0., -0.1, 0.8], [0., 0., 0.]])
IS_SHOWING_2DVISUALIZATION = True

# Proportional gain for both controllers
RADIUS = 0.21
MAX_VEL = 0.5
MAX_ROT = 5
SAFE_DISTANCE = 0.51
Rsi = 0.51

# Proportional gain for both controllers
GAIN = 5
b = 10
order = 1

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)


# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state):
    h = np.zeros(3)
    H = np.zeros((3, 2))
    obs_distances = np.zeros(3)

    # H matrix construction
    for i in range(np.shape(obstacles)[1]):
        obs_distance = np.linalg.norm(robot_state[0:2] - obstacles[0:2, i])
        h[i] = np.linalg.norm(robot_state[0:2] - obstacles[0:2, i]) ** 2 - Rsi ** 2
        H[i, :] = 2 * (robot_state[0:2] - obstacles[0:2, i])
        obs_distances[i] = obs_distance

    # Gamma calculation
    h = b * (h ** order)

    # Regulated u_gtg
    u_gtg = saturate_velocity(GAIN * (desired_state - robot_state))

    # Construct Q, H, b, c for QP-based controller
    Q_mat = 2 * cvxopt.matrix(np.eye(2), tc='d')
    c_mat = -2 * cvxopt.matrix(u_gtg[:2], tc='d')

    H_mat = cvxopt.matrix(-H, tc='d')
    b_mat = cvxopt.matrix(h, tc='d')

    # Find u*
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(Q_mat, c_mat, H_mat, b_mat, verbose=False)

    current_input = np.array([sol['x'][0], sol['x'][1], 0])
    return saturate_velocity(current_input), obs_distances, u_gtg


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
    desired_state = np.array([2., 1., 0.])  # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, 4))  # for [vx, vy, vx_gtg, vy_gtg] vs iteration time
    obs_history = np.zeros((sim_iter, 3))

    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_visualizer = sim_mobile_robot('omnidirectional')  # Omnidirectional Icon
        # sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        for i in range(np.shape(obstacles)[1]):
            sim_visualizer.ax.add_patch(plt.Circle(obstacles[0:2, i], 0.3, color='r'))
            sim_visualizer.ax.add_patch(plt.Circle(obstacles[0:2, i], SAFE_DISTANCE, color='r', fill=False))
        sim_visualizer.show_goal(desired_state)

    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # COMPUTE CONTROL INPUT
        # ------------------------------------------------------------
        current_input, obs_distances, u_gtg = compute_control_input(desired_state, robot_state)
        # ------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = np.append(current_input[0:2], u_gtg[0:2])
        obs_history[it, :] = obs_distances

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
    ax[0].plot(t, input_history[:, 2], label='vx_gtg [m/s]')
    ax[0].plot(t, input_history[:, 3], label='vy_gtg [m/s]')
    ax[1].plot(t, np.sqrt(input_history[:, 0] ** 2 + input_history[:, 1] ** 2), label='v [m/s]')
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

    # Plot historical errors and distance to obstacles
    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, goal_history[:, 0] - state_history[:, 0], label='error in x [m]')
    ax.plot(t, goal_history[:, 1] - state_history[:, 1], label='error in y [m]')
    ax.plot(t, obs_history[:, 0], label='distance to obstacle 1 [m]')
    ax.plot(t, obs_history[:, 1], label='distance to obstacle 2 [m]')
    ax.plot(t, obs_history[:, 2], label='distance to obstacle 3 [m]')
    ax.set(xlabel="t [s]", ylabel="distance")
    plt.legend()
    plt.grid()

    # Plot historical h values
    fig5 = plt.figure(5)
    ax = plt.gca()
    ax.plot(t, obs_history[:, 0] ** 2 - Rsi ** 2, label='function 1 [m]')
    ax.plot(t, obs_history[:, 1] ** 2 - Rsi ** 2, label='function 2 [m]')
    ax.plot(t, obs_history[:, 2] ** 2 - Rsi ** 2, label='function 3 [m]')
    ax.set(xlabel="t [s]", ylabel="value")
    plt.legend()
    plt.grid()

    plt.show()
