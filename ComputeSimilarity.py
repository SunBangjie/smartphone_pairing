from matplotlib import pyplot as plt
import numpy as np
import Threshold


def sum_3d(a, b, m):
    c = [0, 0, 0]
    c[0] = a[0] + b[0] * m
    c[1] = a[1] + b[1] * m
    c[2] = a[2] + b[2] * m
    return c


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


def plot(verifier, sender, attacker):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    # Plot verifier
    lists = sorted(verifier.items())
    x, y = zip(*lists)
    ax1.plot(x, y)
    ax1.set_title("verifier")
    # Plot sender
    lists = sorted(sender.items())
    x, y = zip(*lists)
    ax2.plot(x, y)
    ax2.set_title("legitimate")
    # Plot attacker
    lists = sorted(attacker.items())
    x, y = zip(*lists)
    ax3.plot(x, y)
    ax3.set_title("attacker")

    # Show plots
    plt.tight_layout()
    plt.show()


def find_keys_in_range(keys, min, max):
    result = []
    for key in keys:
        if min <= key <= max:
            result.append(key)
    return result


def get_similarity(a, b):
    same_sign = a * b > 0
    small_diff = abs(abs(a) - abs(b)) < Threshold.DIFF_THRESHOLD
    if same_sign and small_diff:
        return 1
    elif same_sign:
        return 0.9
    elif small_diff:
        return 0.8
    else:
        return 0


def get_similarity_3d(a, b):
    similarity = 0
    for i in [0, 1, 2]:
        similarity = similarity + get_similarity(a[i], b[i])
    return similarity / 3


def main(experiment_name, LOG=False):
    # get file paths
    folder = "Experiment_Output/{}/".format(experiment_name)
    sender_acc_path = folder + "{}_acc_reading.txt".format(experiment_name)
    attacker_acc_path = folder + \
        "{}_attack_acc_reading.txt".format(experiment_name)
    verifier_vel_path = folder + "velocities.txt"

    # load data
    first_ts, last_ts = get_first_and_last_ts(verifier_vel_path)
    verifier_vel = read_acc(verifier_vel_path, first_ts,
                            last_ts, is_sender=False)
    sender_acc = read_acc(sender_acc_path, first_ts, last_ts, is_sender=True)
    attacker_acc = read_acc(attacker_acc_path, first_ts,
                            last_ts, is_sender=True)

    # log number of samples
    if LOG:
        print("Verifier (Kinect) has {} samples".format(len(verifier_vel.keys())))
        print("Sender (Smartphone) has {} samples".format(len(sender_acc.keys())))
        print("Attacker (Malicious phone) has {} samples".format(
            len(attacker_acc.keys())))

    # visualize
    # plot(verifier_vel, sender_acc, attacker_acc)

    # since we always have more samples in sender/attacker, we want to sum the accelerations within the time interval to get the next velocity
    verifier_ts = list(verifier_vel.keys())
    sender_vel = {}
    attacker_vel = {}
    for i in range(len(verifier_ts) - 1):
        min = verifier_ts[i]
        max = verifier_ts[i+1]
        sender_ts = find_keys_in_range(list(sender_acc.keys()), min, max)
        sum = [0, 0, 0]
        for ts in sender_ts:
            sum = sum_3d(sum, sender_acc[ts], (ts - min) / 1000)
        sender_vel[max] = sum
        attacker_ts = find_keys_in_range(list(attacker_acc.keys()), min, max)
        sum = [0, 0, 0]
        for ts in attacker_ts:
            sum = sum_3d(sum, attacker_acc[ts], (ts - min) / 1000)
        attacker_vel[max] = sum

    verifier_vel = normalize_3d(verifier_vel)
    sender_vel = normalize_3d(sender_vel)
    attacker_vel = normalize_3d(attacker_vel)

    # visualize
    plot(verifier_vel, sender_vel, attacker_vel)

    # get similarity in each axis
    legit_sum = 0
    attacker_sum = 0
    legit_sim = {}
    attacker_sim = {}
    for i in range(len(verifier_ts) - 1):
        ts = verifier_ts[i+1]
        legit_sum = legit_sum + \
            get_similarity_3d(verifier_vel[ts], sender_vel[ts])
        legit_sim[ts] = legit_sum / (i+1)
        attacker_sum = attacker_sum + get_similarity_3d(
            verifier_vel[ts], attacker_vel[ts])
        attacker_sim[ts] = attacker_sum / (i+1)

    print("Legitimate similarity: {}".format(
        round(legit_sum / len(verifier_ts) * 100, 2)))
    print("Attacker similarity: {}".format(
        round(attacker_sum / len(verifier_ts) * 100, 2)))

    # visualize
    # plot(verifier_vel, legit_sim, attacker_sim)


if __name__ == "__main__":
    # "exp3" - "exp6" involve attacker
    for i in [3, 4, 5, 6]:
        experiment_name = "exp{}".format(i)
        print("Doing Experiment {}:".format(i))
        main(experiment_name)
        print("")
