# Fundamental of Mobile Robot - AUT.710 - 2023
# Exercise 05 - Problem 02
# Hoang Pham, Nadun Ranasinghe

import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot
from library.detect_obstacle import DetectObstacle

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 30 # total simulation duration in seconds
# Set initial state
init_state = np.array([-2., 1., 0.])  # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Robot and environment constraints
RADIUS = 0.21
MAX_VEL = 0.5
MAX_ROT = 5
SAFE_DISTANCE = 0.4
EPSILON = 0.1

# Proportional gain for controllers
GAIN = 5
ALPHA_1 = 0.2
ALPHA_2 = 0.8
DESIRED_DISTANCE = SAFE_DISTANCE + .8 * EPSILON


# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)

# Define Obstacles 
obst_vertices = np.array([[-1., 1.2], [-1., 0.8], [0., 0.8], [0.5, 0.5], [0.5, -0.5], [0., -0.8], \
                          [-1, -0.8], [-1., -1.2], [1., -1.2], [1., 1.2], [-1., 1.2]])
c_rotation = np.array([[0, 1], [-1, 0]])
cc_rotation = np.array([[0, -1], [1, 0]])

# Define sensor's sensing range and resolution
sensing_range = 1.  # in meter
sensor_resolution = np.pi / 8  # angle between sensor data in radian


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
    # Variables to evaluate the conditions
    min_distance = np.min(distance_reading)
    min_sensor = np.argmin(distance_reading)
    min_coordinate = obst_points[:, min_sensor]
    goal_distance = calculate_distance(desired_state, robot_state)
    goal_direction = (desired_state - robot_state)[0:2]
    avoid_direction = (robot_state[0:2] - min_coordinate)

    # print(avoid_direction, goal_direction, c_rotation @ avoid_direction)

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
        print(policy, robot_state[0], robot_state[1])
    elif (policy == 'gtg' or policy == 'avo') and condition_1 and condition_3:
        switch_range = goal_distance
        policy = 'wf-cc'
        print(policy, robot_state[0], robot_state[1])
    elif (policy == 'wf-c' or policy == 'wf-cc') and condition_4 and condition_5:
        policy = 'gtg'
        print(policy, robot_state[0], robot_state[1])
    elif (policy == 'wf-c' or policy == 'wf-cc') and condition_6:
        policy = 'avo'
        print(policy, robot_state[0], robot_state[1])

    # print(policy, switch_range, goal_distance, goal_distance - switch_range)

    # Compute the control input
    if policy == 'gtg':
        current_input = GAIN * (desired_state - robot_state)

    elif policy == 'wf-c' or policy == 'wf-cc':
        # Picking a pair of two sensors those are closest to the obstacle
        c_sensor = min_sensor - 1
        c_sensor = c_sensor + 16 if c_sensor < 0 else c_sensor

        cc_sensor = min_sensor + 1
        cc_sensor = cc_sensor - 16 if cc_sensor > 15 else cc_sensor

        cc_wise_sensor = True if distance_reading[c_sensor] > distance_reading[cc_sensor] else False
        neighbor_coordinate = obst_points[:, cc_sensor] if cc_wise_sensor else obst_points[:, c_sensor]

        # Calculate the tangential movement
        u_wf_t = neighbor_coordinate - min_coordinate
        u_wf_t = u_wf_t if (policy == 'wf-c' and cc_wise_sensor) or (
                policy == 'wf-cc' and not cc_wise_sensor) else -u_wf_t
        u_wf_t = u_wf_t / np.linalg.norm(u_wf_t)

        # Calculate the perpendicular movement
        u_wf_p = min_coordinate - robot_state[0:2] - np.dot(min_coordinate - robot_state[0:2], u_wf_t) * u_wf_t
        u_wf_p = u_wf_p * (1 - DESIRED_DISTANCE / np.linalg.norm(u_wf_p))

        # The control input consists two movements
        current_input = ALPHA_1 * u_wf_p + ALPHA_2 * u_wf_t
        current_input = np.hstack((current_input, np.array([0.])))

    else:
        # Collision avoidance policy
        current_input = GAIN * (robot_state - np.hstack((min_coordinate, 0.)))

    return saturate_velocity(current_input), policy, switch_range


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
    desired_state = np.array([2., 0., 0.])  # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, 3))  # for [vx, vy, omega] vs iteration time
    sensor_history = np.zeros(sim_iter)

    # Initiate the Obstacle Detection
    range_sensor = DetectObstacle(sensing_range, sensor_resolution)
    range_sensor.register_obstacle_bounded(obst_vertices)
    switch_range = 0.  # Initialize the switch range
    policy = 'gtg'  # policy initialization

    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_visualizer = sim_mobile_robot('omnidirectional')  # Omnidirectional Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.show_goal(desired_state)

        # Display the obstacle
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

    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # Get information from sensors
        distance_reading = range_sensor.get_sensing_data(robot_state[0], robot_state[1], robot_state[2])
        obst_points = compute_sensor_endpoint(robot_state, distance_reading)
        # print(distance_reading)
        # print(obst_points)

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
            for i in range(len(distance_reading)): pl_txt[i].set_position((obst_points[0, i], obst_points[1, i]))

        # --------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        robot_state = robot_state + Ts * current_input  # will be used in the next iteration
        robot_state[2] = ((robot_state[2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

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

    # Plot historical errors and distance to goal
    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, goal_history[:, 0] - state_history[:, 0], label='error in x [m]')
    ax.plot(t, goal_history[:, 1] - state_history[:, 1], label='error in y [m]')
    ax.set(xlabel="t [s]", ylabel="distance")
    plt.legend()
    plt.grid()

    # Plot historical minimum measurement of sensor
    fig5 = plt.figure(5)
    ax = plt.gca()
    ax.plot(t, sensor_history, label='minimal measurement [m]')
    ax.set(xlabel="t [s]", ylabel="distance")
    plt.legend()
    plt.grid()

    plt.show()
