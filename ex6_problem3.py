# Fundamental of Mobile Robot - AUT.710 - 2023
# Exercise 06 - Problem 03
# Hoang Pham, Nadun Ranasinghe

import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot
# from ex6p3_obstacles import dict_obst_vertices
from ex6p3_obstacles import dict_obst_hard_mode as dict_obst_vertices
import cvxopt

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 300  # total simulation duration in seconds
# Set initial state
init_state = np.array([-4., -3.5, 0.])  # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Robot model
ROBOT_RADIUS = 0.21
WHEEL_RADIUS = 0.1
MAX_SPEED = 10.

# Controller selection
ALTERNATIVE_CONTROL_POINT = True
TIME_VARYING_GAIN = True

# Proportional gain for controllers
GAIN = 2.0
TOLERANCE = 0.2

# Define Field size for plotting (should be in tuple)
field_x = (-5, 5)
field_y = (-4, 4)

# Alternative control point
l = ROBOT_RADIUS * 0.5
Rsi = ROBOT_RADIUS + l
SAFE_DISTANCE = Rsi
DETECT_RANGE = 1.0

# Obstacles
obstacles = np.empty((0, 3))
SCALE_RSI = False       # To generate roof-shape boundary
RESOLUTION = 2.5        # To generate point obstacle

# Gamma set
b = 10
order = 3

# Goals
DESIRED_STATES = np.array([[4., 0., 0.], [-0.5, 3.7, 0.], [-4, -3.5, 0.]])

# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state):
    # initial numpy array for [vlin, omega, wl, wr]
    all_input = np.array([0., 0., 0., 0.])

    # POI
    theta = robot_state[2]
    interest = robot_state[0:2] + np.array([l * np.cos(theta), l * np.sin(theta)])

    # Initialize constraints
    num_obs = np.shape(obstacles)[0]
    h = np.empty((0, 1))
    H = np.empty((0, 2))
    distance = calculate_distance(desired_state, interest)
    min_obst_dist = DETECT_RANGE

    # H matrix construction
    for i in range(num_obs):
        obs_distance = calculate_distance(interest, obstacles[i, :2])
        if obs_distance < DETECT_RANGE:
            h = np.vstack((h, obs_distance ** 2 - obstacles[i, 2] ** 2))
            H = np.vstack((H, 2 * (interest - obstacles[i, :2])))
            min_obst_dist = obs_distance if obs_distance < min_obst_dist else obs_distance

    # Gamma calculation
    h = b * (h ** order)

    # Proportional gain computation
    if TIME_VARYING_GAIN:
        K = GAIN * (1 - np.exp(-distance)) / distance
    else:
        K = GAIN

    u_gtg = K * (desired_state[0:2] - interest)

    # Construct Q, H, b, c for QP-based controller
    Q_mat = 2 * cvxopt.matrix(np.eye(2), tc='d')
    c_mat = -2 * cvxopt.matrix(u_gtg[:2], tc='d')
    H_mat = cvxopt.matrix(-H, tc='d')
    b_mat = cvxopt.matrix(h, tc='d')

    # Find u*
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(Q_mat, c_mat, H_mat, b_mat, verbose=False)

    # Transformation matrix
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    transformation = np.array([[1, 0], [0, 1 / l]]) @ rotation

    # Origin control input
    u_gtg = transformation @ np.array([sol['x'][0], sol['x'][1]])

    # Saturation
    all_input[0], all_input[1], all_input[2], all_input[3] = saturate(u_gtg[0], u_gtg[1])

    return all_input, min_obst_dist


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


def generate_point_obstacles():
    global obstacles

    # Iterate through list of obstacles
    for obst_vertices in dict_obst_vertices.values():

        # Iterate through edges
        for i in range(len(obst_vertices)):
            v1 = obst_vertices[i]
            v2 = obst_vertices[i + 1] if i < len(obst_vertices) - 1 else obst_vertices[0]
            res = np.ceil(RESOLUTION * calculate_distance(v2, v1) / Rsi)

            # Generate point obstacles on the edges
            for j in range(int(res)):
                scale = (3 - 2 * abs(j / res - 1/2)) / 2 if SCALE_RSI else 1.
                obstacles = np.vstack((obstacles, np.hstack(((v2 - v1) * j / res + v1, Rsi * scale))))


# MAIN SIMULATION COMPUTATION
# ---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max / Ts)  # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy()  # numpy array for [px, py, theta]
    goal_id = 0
    desired_state = DESIRED_STATES[goal_id]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, 4))     # for [v, w, wl, wr] vs iteration time
    obs_history = np.zeros((sim_iter, 1))

    # Make point obstacles
    generate_point_obstacles()

    # Initialize the visualization
    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_visualizer = sim_mobile_robot('unicycle') # Unicycle Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.show_goal(desired_state)

        # Point obstacles
        for i in range(np.shape(obstacles)[0]):
            sim_visualizer.ax.add_patch(plt.Circle(obstacles[i, :2], obstacles[i, 2], color='r', fill=False))
        sim_visualizer.show_goal(desired_state)

        # Edges
        for obst_vertices in dict_obst_vertices.values():
            sim_visualizer.ax.plot(obst_vertices[:, 0], obst_vertices[:, 1], '--r')

    # Simulate
    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # COMPUTE CONTROL INPUT
        # ------------------------------------------------------------
        current_input, min_obst_dist = compute_control_input(desired_state, robot_state)
        # ------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input
        obs_history[it] = min_obst_dist

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(current_time)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data

        # --------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        theta = robot_state[2]
        B = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
        robot_state = robot_state + Ts * (B @ current_input[0:2])  # will be used in the next iteration
        robot_state[2] = ((robot_state[2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

        # Update desired state if the current goal is reached
        # print(goal_id, it, calculate_distance(desired_state, robot_state))
        if calculate_distance(desired_state, robot_state) < TOLERANCE:
            goal_id = goal_id if goal_id == len(DESIRED_STATES) - 1 else goal_id + 1
            desired_state = DESIRED_STATES[goal_id]
            sim_visualizer.update_goal(desired_state)

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
    ax.plot(t, goal_history[:, 0], ':', label='goal px [m]')
    ax.plot(t, goal_history[:, 1], ':', label='goal py [m]')
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

    # Plot measurement
    fig5 = plt.figure(5)
    ax = plt.gca()
    ax.plot(t, obs_history, label='minimum measurement [m]')
    ax.set(ylabel="measurement", xlabel="t [s]")
    plt.legend()
    plt.grid()

    plt.show()