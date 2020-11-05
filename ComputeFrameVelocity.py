import numpy as np

experiment_name = "exp1"
output_folder = "Experiment_Output/" + experiment_name + "/"
f = open(output_folder + "positions.txt", "r")
out = open(output_folder + "velocities.txt", "w")

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
    velocity = (pos - prev_pos) / (time - prev_time)
    out.write("{},{},{},{}\n".format(
        time, velocity[0], velocity[1], velocity[2]))

    # step
    prev_time = time
    prev_pos = pos

f.close()
out.close()
