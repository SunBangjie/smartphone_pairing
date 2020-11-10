from matplotlib import pyplot as plt
import numpy as np
import Threshold


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def moving_average_3d(d):
    T = list(d.keys())
    X = []
    Y = []
    Z = []
    for key in T:
        acc = d[key]
        X.append(acc[0])
        Y.append(acc[1])
        Z.append(acc[2])
    N = Threshold.MOVING_AVERAGE_N
    X = moving_average(np.array(X), n=N)
    Y = moving_average(np.array(Y), n=N)
    Z = moving_average(np.array(Z), n=N)
    result = {}
    for i in range(len(X)):
        result[i] = [X[i], Y[i], Z[i]]
    return result


def sum_3d(a, b, m):
    c = [0, 0, 0]
    c[0] = a[0] + b[0] * m
    c[1] = a[1] + b[1] * m
    c[2] = a[2] + b[2] * m
    return c


def scale(a, m):
    for i in [0, 1, 2]:
        a[i] = a[i] * m
    return a


def scale_axis(d, m, axis):
    result = {}
    for key, value in d.items():
        value[axis] = value[axis] * m
        result[key] = value
    return result


def step(a):
    a[0] = 1 if a[0] > 0 else 0
    a[1] = 1 if a[1] > 0 else 0
    a[2] = 1 if a[2] > 0 else 0
    return a


def step_3d(d):
    values = np.array(list(d.values()))
    result = {}
    for key, value in d.items():
        result[key] = step(value)
    return result


def maxabs(a, axis=None):
    """Return slice of a, keeping only those values that are furthest away
    from 0 along axis"""
    maxa = a.max(axis=axis)
    mina = a.min(axis=axis)
    p = abs(maxa) > abs(mina)  # bool, or indices where +ve values win
    n = abs(mina) > abs(maxa)  # bool, or indices where -ve values win
    if axis == None:
        if p:
            return maxa
        else:
            return mina
    shape = list(a.shape)
    shape.pop(axis)
    out = np.zeros(shape, dtype=a.dtype)
    out[p] = maxa[p]
    out[n] = mina[n]
    return out


def normalize_3d(d):
    values = np.array(list(d.values()))
    max = maxabs(values, axis=0)
    result = {}
    for key, value in d.items():
        result[key] = [value[0]/max[0], value[1]/max[1], value[2]/max[2]]
    return result


def swap_axis(a, a1, a2):
    temp = a[a1]
    a[a1] = a[a2]
    a[a2] = temp
    return a


def swap_axis_3d(d, a1, a2):
    result = {}
    for key, value in d.items():
        result[key] = swap_axis(value, a1, a2)
    return result


def cap(a, m):
    for i in [0, 1, 2]:
        if a[i] > m:
            a[i] = m
        elif a[i] < -m:
            a[i] = -m
    return a


def cap_3d(d, m):
    result = {}
    for key, value in d.items():
        result[key] = cap(value, m)
    return result


def get_first_and_last_ts(verifier_filename):
    first_ts = 0
    last_ts = 0
    with open(verifier_filename, 'r') as f:
        lines = f.readlines()
        first_ts = int(lines[0].split(',')[0])
        last_ts = int(lines[-1].split(',')[0])
    return first_ts, last_ts


def read_acc(filename, first_ts, last_ts, is_sender=True):
    result = {}  # {timestamp: [x, y, z]}
    with open(filename, 'r') as f:
        for line in f.readlines():
            if is_sender:
                timestamp = int(line.split(':')[0])
                acc = line.split(':')[1]
            else:
                timestamp = int(line.split(',')[0])
                acc_list = line.split(',')[1:]
                acc = ','.join(acc_list)
            # check valid range
            if first_ts <= timestamp <= last_ts:
                # key is relative timestamp
                result[(timestamp - first_ts)
                       ] = list(map(lambda x: float(x.strip()), acc.split(',')))
    return result


def plot_sender(verifier, sender, title):
    fig, (ax1, ax2) = plt.subplots(2)
    plt.title(title)
    # Plot verifier
    lists = sorted(verifier.items())
    x, y = zip(*lists)
    ax1.plot(x, y)
    ax1.set_title("verifier")
    # Plot sender
    lists = sorted(sender.items())
    x, y = zip(*lists)
    ax2.plot(x, y)
    ax2.set_title("sender")
    # Show plots
    plt.tight_layout()
    plt.show()


def plot_correlation(X, Y, Z, title):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    T = range(len(X))
    ax1.plot(T, X)
    ax1.set_title("X axis")
    ax2.plot(T, Y)
    ax2.set_title("Y axis")
    ax3.plot(T, Z)
    ax3.set_title("Z axis")
    plt.tight_layout()
    plt.title(title)
    plt.show()


def find_keys_in_range(keys, min, max):
    result = []
    for key in keys:
        if min <= key <= max:
            result.append(key)
    return result


def get_axis_similarity(a, b):
    same_sign = a * b > 0
    small_diff = abs(a - b) < Threshold.DIFF_THRESHOLD
    if same_sign and small_diff:
        return 1
    elif same_sign:
        return 0.8
    elif small_diff:
        return 0.5
    return 0


def get_distance(a, b):
    sum = 0
    for i in [0, 1, 2]:
        sum = sum + (a[i] - b[i]) ** 2
    return sum ** 0.5


def get_similarity_3d(a, b):
    similarity = 0
    # get average axis similarity
    for i in [0, 1, 2]:
        similarity = similarity + get_axis_similarity(a[i], b[i])
    similarity = similarity / 3
    # get inverse of euclidean distance
    similarity = similarity + 1 / (get_distance(a, b) + 1)
    return similarity


def align_sampling_rates(verifier_acc, sender_acc):
    # since we always have more samples in sender/attacker, we want to average the accelerations within the time interval
    verifier_ts = list(verifier_acc.keys())
    sender_vel = {}
    for i in range(len(verifier_ts) - 1):
        min = verifier_ts[i]
        max = verifier_ts[i+1]
        sender_ts = find_keys_in_range(list(sender_acc.keys()), min, max)
        sum = [0, 0, 0]
        counter = 0
        for ts in sender_ts:
            sum = sum_3d(sum, sender_acc[ts], 1)
            counter = counter + 1
        if counter > 0:
            sender_vel[max] = scale(sum, 1 / counter)
    return sender_vel


def correlate_3d(a, b, mode):
    aX, aY, aZ = a[:, 0], a[:, 1], a[:, 2]
    bX, bY, bZ = b[:, 0], b[:, 1], b[:, 2]
    simX, simY, simZ = np.correlate(aX, bX, mode), np.correlate(
        aY, bY, mode), np.correlate(aZ, bZ, mode)
    return simX, simY, simZ


def trim_samples(d):
    size = len(list(d.keys()))
    head = int(size * Threshold.HEAD)
    tail = int(size * Threshold.TAIL)
    counter = 0
    result = {}
    for key in d.keys():
        counter = counter + 1
        if head <= counter <= tail and counter <= Threshold.NUM_SAMPLES:
            result[key] = d[key]
    return result


def compute_simularity(experiment_name, LOG=False, is_attacker=False, PLOT=False, normalize_samples=True):
    # get file paths
    folder = "Experiment_Output/{}/".format(experiment_name)
    data_folder = "Experiment_Data/{}/".format(experiment_name)
    verifier_acc_path = folder + "accelerations.txt"
    if is_attacker:
        sender_acc_path = data_folder + \
            "{}_attack_acc_reading.txt".format(experiment_name)
    else:
        sender_acc_path = data_folder + \
            "{}_acc_reading.txt".format(experiment_name)

    # get identity
    if is_attacker:
        identity = "Attacker"
    else:
        identity = "Legitimate"

    # load data
    first_ts, last_ts = get_first_and_last_ts(verifier_acc_path)
    verifier_acc = read_acc(verifier_acc_path, first_ts,
                            last_ts, is_sender=False)
    sender_acc = read_acc(sender_acc_path, first_ts, last_ts, is_sender=True)

    # log number of samples
    if LOG:
        print("Verifier (Kinect) has {} samples".format(len(verifier_acc.keys())))
        print("{} (Smartphone) has {} samples".format(
            identity, len(sender_acc.keys())))

    # trim samples
    verifier_acc = trim_samples(verifier_acc)

    # visualize
    if PLOT:
        plot_sender(verifier_acc, sender_acc,
                    "Raw Acceleration with {}".format(identity))

    # align sampling rate
    sender_vel = align_sampling_rates(verifier_acc, sender_acc)

    # take moving average
    verifier_acc = moving_average_3d(verifier_acc)
    sender_vel = moving_average_3d(sender_vel)

    # normalize
    verifier_acc = normalize_3d(verifier_acc)
    sender_vel = normalize_3d(sender_vel)

    # flip x and z-axis
    verifier_acc = scale_axis(verifier_acc, -1, 2)
    verifier_acc = scale_axis(verifier_acc, -1, 0)

    # visualize
    if PLOT:
        plot_sender(verifier_acc, sender_vel,
                    "Processed Acceleration with {}".format(identity))

    simX, simY, simZ = correlate_3d(
        np.array(list(verifier_acc.values())),
        np.array(list(sender_vel.values())),
        'same')

    # take absolute
    simX, simY, simZ = np.abs(simX), np.abs(simY), np.abs(simZ)

    # visualize
    if PLOT:
        plot_correlation(
            simX, simY, simZ, "Absolute of cross correlation coefficients in 3 axes with {}".format(identity))

    # Compute overall similarity
    simMean = np.mean(simX) + np.mean(simY) + np.mean(simZ)
    simMean = simMean / 3

    if normalize_samples:
        # Normalize similarity with number of samples as well
        simMean = simMean / len(verifier_acc.keys()) * 100

    return round(simMean, 4)
