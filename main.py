import VerifierCameraReader
import PhoneDetector
import VisualizePositions
import ComputeAndVisualizeAcceleration
import ComputeSimilarity
import Utils.ConvertFramesToVideo

import time
from matplotlib import pyplot as plt
import numpy as np


def plot_round_result(result):
    # plot round result
    barWidth = 0.3
    bars1 = result[3]
    bars2 = result[5]
    bars3 = result[7]
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, color='#7f6d5f', width=barWidth,
            edgecolor='white', label='n_rounds = 3')
    plt.bar(r2, bars2, color='#557f2d', width=barWidth,
            edgecolor='white', label='n_rounds = 5')
    plt.bar(r3, bars3, color='#2d7f5e', width=barWidth,
            edgecolor='white', label='n_rounds = 7')

    # Add xticks on the middle of the group bars
    plt.xlabel('type of experiment', fontweight='bold')
    plt.ylabel('similarity score', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))],
               ['A', 'B', 'C', 'D', 'E', 'F', 'G'])

    # Create legend & Show graphic
    plt.legend()
    plt.show()


def extract_data(experiment_name):
    # extract frames
    VerifierCameraReader.extract_frames(experiment_name)

    # extract RGBD data from frames
    PhoneDetector.detect(experiment_name)

    # visualize positions
    # VisualizePositions.visualize_position(experiment_name)


def do_experiment_with_legitimate(experiment_name, normalize=True):
    # compute and visualize acceleration
    ComputeAndVisualizeAcceleration.compute(experiment_name)

    # compute similarity
    legit_sim = ComputeSimilarity.compute_simularity(
        experiment_name, normalize_samples=normalize)

    return legit_sim


def experiment_type_1(normalize=True):
    exp_list = range(1, 22)
    exp_type_label = {
        "A": "only horizontal movement",
        "B": "only vertical movement",
        "C": "only front and back movement",
        "D": "movement in XY-plane",
        "E": "movement in XZ-plane",
        "F": "movement in YZ-plane",
        "G": "movement in XYZ 3D space"
    }

    result = {
        # round_number: exp_similarity_list
    }

    for i in exp_list:
        round_number = (i-1) % 3 * 2 + 3
        exp_type = (i-1) // 3
        experiment_name = "exp{}".format(i)

        # conduct experiment type 1
        similarity = do_experiment_with_legitimate(
            experiment_name, normalize=normalize)

        # store round result
        if round_number in result:
            result[round_number].append(similarity)
        else:
            result[round_number] = [similarity]

    plot_round_result(result)


def do_experiment_with_attacker(experiment_name, normalize=True):
    # compute and visualize acceleration
    ComputeAndVisualizeAcceleration.compute(experiment_name)

    # compute similarity
    legit_sim = ComputeSimilarity.compute_simularity(
        experiment_name, normalize_samples=normalize)
    attack_sim = ComputeSimilarity.compute_simularity(
        experiment_name, is_attacker=True, normalize_samples=normalize)

    return legit_sim, attack_sim


def experiment_type_2(normalize=True):
    exp_list = range(22, 34)
    Bob = []
    Mally = []
    for i in exp_list:
        experiment_name = "exp{}".format(i)
        legit_sim, attack_sim = do_experiment_with_attacker(
            experiment_name, normalize=normalize)
        Bob.append(legit_sim)
        Mally.append(attack_sim)

    # plot round result
    barWidth = 0.3
    bars1 = Bob
    bars2 = Mally

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, bars1, color='#32a852', width=barWidth,
            edgecolor='white', label='Legitimate')
    plt.bar(r2, bars2, color='#f04747', width=barWidth,
            edgecolor='white', label='Attacker')

    # Add xticks on the middle of the group bars
    plt.xlabel('type of experiment', fontweight='bold')
    plt.ylabel('similarity score', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))],
               ['A~0', 'B~0', 'C~0', 'D~0', 'A~45', 'B~45', 'C~45', 'D~45', 'A~90', 'B~90', 'C~90', 'D~90'])

    # Create legend & Show graphic
    plt.legend()
    plt.show()


def experiment_type_3(normalize=True):
    exp_list = range(34, 44)
    Bob = []
    Mally = []
    for i in exp_list:
        experiment_name = "exp{}".format(i)
        legit_sim, attack_sim = do_experiment_with_attacker(
            experiment_name, normalize=normalize)
        Bob.append(legit_sim)
        Mally.append(attack_sim)

    # plot round result
    barWidth = 0.3
    bars1 = Bob
    bars2 = Mally

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, bars1, color='#32a852', width=barWidth,
            edgecolor='white', label='Legitimate')
    plt.bar(r2, bars2, color='#f04747', width=barWidth,
            edgecolor='white', label='Attacker')

    # Add xticks on the middle of the group bars
    plt.xlabel('experiments', fontweight='bold')
    plt.ylabel('similarity score', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))],
               range(1, 11))

    # Create legend & Show graphic
    plt.legend()
    plt.show()

    T = []
    F = []
    for threshold in range(0, 100, 1):
        TPR = 0
        FPR = 0
        for b in Bob:
            if b > threshold / 10:
                TPR = TPR + 1
        for m in Mally:
            if m > threshold / 10:
                FPR = FPR + 1
        TPR = TPR / len(Bob)
        FPR = FPR / len(Mally)
        T.append(TPR)
        F.append(FPR)

    fig = plt.figure()
    # plot points
    plt.scatter(F, T)
    plt.plot(F, T)
    # random guess
    plt.plot([0, 1], [0, 1])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.show()


if __name__ == "__main__":

    # extract data first, if needed
    #exp_list = range(34, 44)
    # for i in exp_list:
    #    experiment_name = "exp{}".format(i)
    #    extract_data(experiment_name)

    experiment_type_1(normalize=False)
    experiment_type_1(normalize=True)
    experiment_type_2(normalize=True)
    experiment_type_3(normalize=True)
