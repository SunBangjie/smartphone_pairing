import numpy as np
import matplotlib.pyplot as plt


def compute(experiment_name):
    output_folder = "Experiment_Output/" + experiment_name + "/"
    f = open(output_folder + "positions.txt", "r")
    out = open(output_folder + "accelerations.txt", "w")
    prev_time = None
    prev_pos = None
    for line in f.readlines():
        split_line = line.split(',')
        if prev_time is None or prev_pos is None:
            prev_time = int(split_line[0])
            prev_pos = np.array([float(split_line[1]), float(
                split_line[2]), float(split_line[3])])
            continue
        else:
            time = int(split_line[0])
            pos = np.array([float(split_line[1]), float(
                split_line[2]), float(split_line[3])])
        # compute v
        velocity = (pos - prev_pos) / ((time - prev_time) ** 2)
        out.write("{},{},{},{}\n".format(
            time, velocity[0], velocity[1], velocity[2]))

        # step
        prev_time = time
        prev_pos = pos

    f.close()
    out.close()


def visualize(experiment_name):
    output_folder = "Experiment_Output/" + experiment_name + "/"
    f = open(output_folder + "accelerations.txt", "r")

    T, X, Y, Z = [], [], [], []

    for line in f.readlines():
        split_line = line.split(',')
        T.append(int(split_line[0][-7:]) / 1000)
        X.append(float(split_line[1]))
        Y.append(float(split_line[2]))
        Z.append(float(split_line[3]))

    plt.plot(T, X, label="X")
    plt.plot(T, Y, label="Y")
    plt.plot(T, Z, label="Depth")
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title("Acceleration graph from RGBD camera")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    experiment_names = ['exp9'] #["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"]
    for name in experiment_names:
        compute(name)
        visualize(name)
