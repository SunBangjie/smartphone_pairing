import numpy as np
import matplotlib.pyplot as plt
import Threshold


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def compute(experiment_name):
    output_folder = "Experiment_Output/" + experiment_name + "/"
    f = open(output_folder + "positions.txt", "r")
    out = open(output_folder + "accelerations.txt", "w")
    prev_time = None
    prev_pos = None
    prev_vel = 0
    last_Z = 0
    for line in f.readlines():
        # Split line to components
        split_line = line.split(',')
        T = int(split_line[0])
        X = float(split_line[1])
        Y = float(split_line[2])
        Z = float(split_line[3])
        if prev_time is None or prev_pos is None:
            prev_time = T
            prev_pos = np.array([X, Y, Z])
            continue
        else:
            time = T
            pos = np.array([X, Y, Z])
        # compute acc
        cur_vel = (2 * (pos - prev_pos)) - prev_vel
        acceleration = (cur_vel - prev_vel) / (time - prev_time)
        # fine tuning Z component of acc
        if Threshold.DEPTH_MIN < Z < Threshold.DEPTH_MAX:
            last_Z = acceleration[2]
        else:
            acceleration[2] = last_Z
        out.write("{},{},{},{}\n".format(
            time, acceleration[0], acceleration[1], acceleration[2]))

        # step
        prev_time = time
        prev_pos = pos
        prev_vel = cur_vel

    f.close()
    out.close()


def visualize_verifier(experiment_name, show_X=True, show_Y=True, show_Z=True, PLOT=True):
    output_folder = "Experiment_Output/" + experiment_name + "/"
    f = open(output_folder + "accelerations.txt", "r")

    T, X, Y, Z = [], [], [], []

    first_line = True
    first_ts = 0
    for line in f.readlines():
        split_line = line.split(',')
        if first_line:
            first_ts = int(split_line[0])
            first_line = False
        T.append(int(split_line[0]) - first_ts)
        X.append(float(split_line[1]))
        Y.append(float(split_line[2]))
        Z.append(float(split_line[3]) * 4)

    N = Threshold.MOVING_AVERAGE_N
    X = moving_average(np.array(X), n=N)
    Y = moving_average(np.array(Y), n=N)
    Z = moving_average(np.array(Z), n=N)

    if show_X:
        plt.plot(range(len(X)), X, label="X")
    if show_Y:
        plt.plot(range(len(Y)), Y, label="Y")
    if show_Z:
        plt.plot(range(len(Z)), Z, label="Depth")
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title("Acceleration graph from RGBD camera")
    plt.legend()
    if PLOT:
        plt.show()


def visualize_sender(experiment_name, show_X=True, show_Y=True, show_Z=True, PLOT=True):
    output_folder = "Experiment_Data/" + experiment_name + "/"
    f = open(output_folder + "{}_acc_reading.txt".format(experiment_name), "r")

    T, X, Y, Z = [], [], [], []

    first_line = True
    first_ts = 0
    for line in f.readlines():
        ts_str = line.split(':')[0]
        accs = (line.split(':')[1]).split(',')
        if first_line:
            first_ts = int(ts_str)
            first_line = False
        T.append(int(ts_str) - first_ts)
        X.append(float(accs[0]))
        Y.append(float(accs[1]))
        Z.append(float(accs[2]))

    N = Threshold.MOVING_AVERAGE_N
    X = moving_average(np.array(X), n=N)
    Y = moving_average(np.array(Y), n=N)
    Z = moving_average(np.array(Z), n=N)

    if show_X:
        plt.plot(range(len(X)), X, label="X")
    if show_Y:
        plt.plot(range(len(Y)), Y, label="Y")
    if show_Z:
        plt.plot(range(len(Z)), Z, label="Depth")
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title("Acceleration graph from accelerometer")
    plt.legend()
    if PLOT:
        plt.show()
