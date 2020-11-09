import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d


def visualize_position(experiment_name):
    output_folder = "Experiment_Output/" + experiment_name + "/"
    f = open(output_folder + "positions.txt", "r")

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
        Z.append(float(split_line[3]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title(experiment_name)
    plt.tight_layout()
    ax.plot3D(X, Y, Z, 'gray')
    ax.scatter3D(X, Y, Z, c=Z, cmap='Greens')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()
