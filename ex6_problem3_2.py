# Fundamental of Mobile Robot - AUT.710 - 2023
# Exercise 06 - Problem 03
# Hoang Pham, Nadun Ranasinghe

import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot
# from ex6p3_obstacles import dict_obst_vertices
from library.detect_obstacle import DetectObstacle
from ex6p3_obstacles import dict_obst_hard_mode as dict_obst_vertices

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 120  # total simulation duration in seconds
# Set initial state
init_state = np.array([-4., -3.5, 0.])  # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Robot model
ROBOT_RADIUS = 0.21
WHEEL_RADIUS = 0.1
MAX_SPEED = 10.


# Controller selection
TIME_VARYING_GAIN = True

# Proportional gain for controllers

# Define Field size for plotting (should be in tuple)
field_x = (-5, 5)
field_y = (-4, 4)

# Alternative control point
l = ROBOT_RADIUS * 0.2

# Controller parameters
Rsi = ROBOT_RADIUS + l
SAFE_DISTANCE = Rsi
EPSILON = 0.2
GAIN = 0.9
ALPHA_1 = 0.7
ALPHA_2 = 0.3
DESIRED_DISTANCE = SAFE_DISTANCE + .5 * EPSILON
TOLERANCE = 0.2

# Obstacles
obst_vertices = np.empty((0, 2))

# Rotation matrix
c_rotation = np.array([[0, 1], [-1, 0]])
cc_rotation = np.array([[0, -1], [1, 0]])

# Define sensor's sensing range and resolution
sensing_range = 1.  # in meter
sensor_num = 16     # per semi-arc
sensor_resolution = np.pi / sensor_num  # angle between sensor data in radian

# Goals
DESIRED_STATES = np.array([[4., 0., 0.], [-0.5, 3.7, 0.], [-4, -3.5, 0.]])

# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_sensor_endpoint(robot_state, sensors_dist):
    # assuming sensor position is in the robot's center
    sens_N = round(2 * np.pi / sensor_resolution)
    sensors_theta = [i * 2 * np.pi / sens_N for i in range(sens_N)]
    obst_points = np.zeros((3, sens_N))

    R_WB = np.array([[np.cos(robot_state[2]), -np.sin(robot_state[2]), robot_state[0]], \
                     [np.sin(robot_state[2]), np.cos(robot_state[2]), robot_state[1]], [0, 0, 1]])

    for i in range(sens_N):
        R_BS = np.array([[np.cos(sensors_theta[i]), -np.sin(sensors_theta[i]), 0], \
                         [np.sin(sensors_theta[i]), np.cos(sensors_theta[i]), 0], [0, 0, 1]])
        temp = R_WB @ R_BS @ np.array([sensors_dist[i], 0, 1])
        obst_points[:, i] = temp

    return obst_points[:2, :]


def compute_control_input(desired_state, robot_state, policy, distance_reading, obst_points, switch_range):
    # initial numpy array for [vlin, omega, wl, wr]
    all_input = np.array([0., 0., 0., 0.])

    # POI
    theta = robot_state[2]
    interest = robot_state[0:2] + np.array([l * np.cos(theta), l * np.sin(theta)])

    # Variables to evaluate the conditions
    min_distance = np.min(distance_reading)
    min_sensor = np.argmin(distance_reading)
    min_coordinate = obst_points[:, min_sensor]
    goal_distance = calculate_distance(desired_state, interest)
    goal_direction = desired_state[:2] - interest
    avoid_direction = (interest - min_coordinate)

    # Evaluate each condition
    condition_1 = True if SAFE_DISTANCE - EPSILON <= min_distance <= SAFE_DISTANCE + EPSILON else False
    condition_2 = True if goal_direction.T @ c_rotation @ avoid_direction > 0 else False
    condition_3 = True if goal_direction.T @ cc_rotation @ avoid_direction > 0 else False
    condition_4 = True if avoid_direction.T @ goal_direction > 0 else False
    condition_5 = True if goal_distance < switch_range else False
    condition_6 = True if min_distance < SAFE_DISTANCE - EPSILON else False

    # Policy switching
    if (policy == 'gtg' or policy == 'avo') and condition_1 and condition_2:
        switch_range = goal_distance
        policy = 'wf-c'
        print(policy, robot_state[0], robot_state[1], switch_range)
    elif (policy == 'gtg' or policy == 'avo') and condition_1 and condition_3:
        switch_range = goal_distance
        policy = 'wf-cc'
        print(policy, robot_state[0], robot_state[1], switch_range)
    elif (policy == 'wf-c' or policy == 'wf-cc') and condition_4 and condition_5:
        policy = 'gtg'
        print(policy)
    elif (policy == 'wf-c' or policy == 'wf-cc') and condition_6:
        policy = 'avo'
        print(policy)

    # Proportional gain computation
    if TIME_VARYING_GAIN:
        K = GAIN * (1 - np.exp(-goal_distance)) / goal_distance
    else:
        K = GAIN

    # Compute the control input
    if policy == 'gtg':
        u_gtg = K * (desired_state[:2] - interest)

    elif policy == 'wf-c' or policy == 'wf-cc':
        # Picking a pair of two sensors those are closest to the obstacle
        c_sensor = min_sensor - 1
        c_sensor = c_sensor + sensor_num * 2 if c_sensor < 0 else c_sensor

        cc_sensor = min_sensor + 1
        cc_sensor = cc_sensor - sensor_num * 2 if cc_sensor > sensor_num - 1 else cc_sensor

        cc_wise_sensor = True if distance_reading[c_sensor] > distance_reading[cc_sensor] else False
        neighbor_coordinate = obst_points[:, cc_sensor] if cc_wise_sensor else obst_points[:, c_sensor]

        # Calculate the tangential movement
        u_wf_t = neighbor_coordinate - min_coordinate
        u_wf_t = u_wf_t if (policy == 'wf-c' and cc_wise_sensor) or (
                policy == 'wf-cc' and not cc_wise_sensor) else -u_wf_t
        u_wf_t = u_wf_t / np.linalg.norm(u_wf_t)

        # Calculate the perpendicular movement
        u_wf_p = min_coordinate - interest - np.dot(min_coordinate - interest, u_wf_t) * u_wf_t
        u_wf_p = u_wf_p * (1 - DESIRED_DISTANCE / np.linalg.norm(u_wf_p))

        # The control input consists two movements
        u_gtg = ALPHA_1 * u_wf_p + ALPHA_2 * u_wf_t

    else:
        # Collision avoidance policy
        u_gtg = K * (interest - min_coordinate)

    # Transformation matrix
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    transformation = np.array([[1, 0], [0, 1 / l]]) @ rotation

    # Origin control input
    u_gtg = transformation @ u_gtg

    # Saturation
    all_input[0], all_input[1], all_input[2], all_input[3] = saturate(u_gtg[0], u_gtg[1])

    return all_input, policy, switch_range


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

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy()  # numpy array for [px, py, theta]
    goal_id = 0
    desired_state = DESIRED_STATES[goal_id]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, 4))
    sensor_history = np.zeros(sim_iter)

    # Initiate the Obstacle Detection
    range_sensor = DetectObstacle(sensing_range, sensor_resolution)

    for obst_vertices in dict_obst_vertices.values():
        range_sensor.register_obstacle_bounded(obst_vertices)

    # Policy initialization
    switch_range = 0.  # Initialize the switch range
    policy = 'gtg'  # policy initialization

    # Initialize the visualization
    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_visualizer = sim_mobile_robot('unicycle') # Unicycle Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.show_goal(desired_state)

        # Display the obstacle
        for obst_vertices in dict_obst_vertices.values():
            sim_visualizer.ax.plot(obst_vertices[:, 0], obst_vertices[:, 1], '--r')

        # get sensor reading
        # Index 0 is in front of the robot.
        # Index 1 is the reading for 'sensor_resolution' away (counter-clockwise) from 0, and so on for later index
        distance_reading = range_sensor.get_sensing_data(robot_state[0], robot_state[1], robot_state[2])

        # compute and plot sensor reading endpoint
        obst_points = compute_sensor_endpoint(robot_state, distance_reading)
        pl_sens, = sim_visualizer.ax.plot(obst_points[0], obst_points[1], '.')  # , marker='X')
        pl_txt = [sim_visualizer.ax.text(obst_points[0, i], obst_points[1, i], str(i)) for i in
                  range(len(distance_reading))]

    # Simulate
    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # Get information from sensors
        distance_reading = range_sensor.get_sensing_data(robot_state[0], robot_state[1], robot_state[2])
        obst_points = compute_sensor_endpoint(robot_state, distance_reading)

        # COMPUTE CONTROL INPUT
        # ------------------------------------------------------------
        current_input, policy, switch_range = compute_control_input(desired_state, robot_state, policy,
                                                                    distance_reading, obst_points, switch_range)
        # ------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input
        sensor_history[it] = np.min(distance_reading)

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(current_time)
            sim_visualizer.update_goal(desired_state)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data
            # update sensor visualization
            pl_sens.set_data(obst_points[0], obst_points[1])
            for i in range(len(distance_reading)):
                pl_txt[i].set_position((obst_points[0, i], obst_points[1, i]))

        # --------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        theta = robot_state[2]
        B = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
        robot_state = robot_state + Ts * (B @ current_input[0:2])  # will be used in the next iteration
        robot_state[2] = ((robot_state[2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

        # Update desired state if the current goal is reached
        if calculate_distance(desired_state, robot_state) < TOLERANCE:
            goal_id = goal_id if goal_id == len(DESIRED_STATES) - 1 else goal_id + 1
            policy = 'gtg'
            desired_state = DESIRED_STATES[goal_id]
            sim_visualizer.update_goal(desired_state)

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history, sensor_history


if __name__ == '__main__':
    # Call main computation for robot simulation
    state_history, goal_history, input_history, sensor_history = simulate_control()

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
    # ax.plot(t, state_history[:, 2], label='theta [rad]')
    ax.plot(t, goal_history[:, 0], ':', label='goal px [m]')
    ax.plot(t, goal_history[:, 1], ':', label='goal py [m]')
    # ax.plot(t, goal_history[:, 2], ':', label='goal theta [rad]')
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
    ax.plot(t, sensor_history, label='minimal measurement [m]')
    ax.set(xlabel="t [s]", ylabel="distance")
    plt.legend()
    plt.grid()

    plt.show()