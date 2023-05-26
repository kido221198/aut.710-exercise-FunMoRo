# Fundamental of Mobile Robot - AUT.710 - 2023
# Exercise 06 - Problem 02
# Hoang Pham, Nadun Ranasinghe

import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 10 * np.pi  # total simulation duration in seconds
# Set initial state
init_state = np.array([0., 0., -np.pi / 2])  # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Robot model
ROBOT_RADIUS = 0.21
WHEEL_RADIUS = 0.1
MAX_SPEED = 10.

# Define controller parameters
K = 0.9
nu = 0.2
alpha = 7.

# Controller selection
POSTURE = True     # False = POSITION
TIME_VARYING_GAIN = True

# Alternative control point
l1 = ROBOT_RADIUS * 0.2
l2 = ROBOT_RADIUS * 0.

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)


# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_control_input(desired_state, feed_forward, robot_state):
    # initial numpy array for [vlin, omega]
    all_input = np.array([0., 0., 0., 0.])
    # Compute the control input
    distance = calculate_distance(desired_state, robot_state)
    theta = robot_state[2]

    # Choosing the controller
    if POSTURE:
        # Dynamic reference
        wd = (feed_forward[3] * feed_forward[0] - feed_forward[2] * feed_forward[1]) / (feed_forward[0] ** 2 + feed_forward[1] ** 2)
        vd = np.sqrt(feed_forward[0] ** 2 + feed_forward[1] ** 2)

        # Error calculation
        A = np.array([[np.cos(theta), np.sin(theta), 0],
                      [-np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        e = A @ (desired_state - robot_state)

        # Posture control input
        u1 = -2 * nu * alpha * e[0]
        u2 = -e[1] * (alpha ** 2 - wd ** 2) / vd - 2 * nu * alpha * e[2]

        # Origin control input
        v = vd * np.cos(e[2]) - u1
        w = wd - u2

    else:
        # Proportional gain computation
        if TIME_VARYING_GAIN:
            # Kv = K * ((1 - abs(angle)/np.pi) ** 1)
            Kv = K * (1 - np.exp(-distance)) / distance
        else:
            Kv = K

        # Transformation matrix
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        H = rotation @ np.array([[1, 0], [0, l1]]) @ np.array([[1, -l2], [0, 1]])
        s_state = robot_state[0:2] + rotation @ np.array([l1, l2])

        # POI control input
        u_gtg = Kv * (desired_state[0:2] - s_state) + feed_forward[0:2]

        # Origin control input
        u_gtg = np.linalg.inv(H) @ u_gtg
        v = u_gtg[0]
        w = u_gtg[1]

    # Saturation
    all_input[0], all_input[1], all_input[2], all_input[3] = saturate(v, w)

    return all_input


def calculate_distance(former, latter):
    # Calculate Euclidean distance
    distance = np.sqrt((former[0] - latter[0]) ** 2 + (former[1] - latter[1]) ** 2)
    return distance.item()


def saturate(v, w):
    # Calculate the wheel velocity
    wr = (2 * v + w * ROBOT_RADIUS) / (2 * WHEEL_RADIUS)
    wl = (2 * v - w * ROBOT_RADIUS) / (2 * WHEEL_RADIUS)

    # Saturate if greater than threshold
    wr = wr if abs(wr) < MAX_SPEED else np.sign(wr) * MAX_SPEED
    wl = wl if abs(wl) < MAX_SPEED else np.sign(wl) * MAX_SPEED

    # Recalculate the control input
    w = (wr - wl) * WHEEL_RADIUS / ROBOT_RADIUS
    v = (wr + wl) * WHEEL_RADIUS / 2

    return v, w, wl, wr


# MAIN SIMULATION COMPUTATION
# ---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max / Ts)  # Total Step for simulation
    desired_trajectory = np.zeros((sim_iter + 1, 3))
    feed_forward = np.zeros((sim_iter + 1, 4))

    # Initialize the trajectory and dynamic properties
    for it in range(sim_iter + 1):
        w = 0.25 * it * Ts
        desired_trajectory[it] = np.array([-2 * np.cos(w), -np.sin(2 * w), np.arctan2(-0.5 * np.cos(2 * w), 0.5 * np.sin(w))])
        feed_forward[it] = np.array([0.5 * np.sin(w), -0.5 * np.cos(2 * w), 0.125 * np.cos(w), 0.25 * np.sin(2 * w)])

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy()  # numpy array for [px, py, theta]
    desired_state = desired_trajectory[0]  # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, 4))  # for [v, w, wl, wr] vs iteration time

    # Initialize the visualization
    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_visualizer = sim_mobile_robot('unicycle')  # Unicycle Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.show_goal(desired_state)

    # Simulate
    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # COMPUTE CONTROL INPUT
        # ------------------------------------------------------------

        current_input = compute_control_input(desired_state, feed_forward[it], robot_state)
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
        robot_state = robot_state + Ts * (B @ current_input[0:2])  # will be used in the next iteration
        robot_state[2] = ((robot_state[2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

        # Update desired state if we consider moving goal position
        desired_state = desired_trajectory[it + 1]

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

    # Plot historical wheels velocity
    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, input_history[:, 2], label='wl [rad/s]')
    ax.plot(t, input_history[:, 3], label='wr [rad/s]')
    ax.set(ylabel="wheels input", xlabel="t [s]")
    plt.legend()
    plt.grid()

    plt.show()
