import matplotlib.pyplot as plt
import numpy as np


f = open("output/velocities.txt", "r")

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
plt.ylabel('Velocity')
plt.title("Velocity graph from RGBD camera")
plt.legend()
plt.show()
