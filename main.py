import VerifierCameraReader
import PhoneDetector
import VisualizePositions
import ComputeAndVisualizeAcceleration
import ComputeSimilarity
import Utils.ConvertFramesToVideo

import time


def extract_data(experiment_name):
    # extract frames
    VerifierCameraReader.extract_frames(experiment_name)

    # extract RGBD data from frames
    PhoneDetector.detect(experiment_name)

    # visualize positions
    VisualizePositions.visualize_position(experiment_name)


def do_experiment(experiment_name, T, LOG=False, PLOT=False):
    # compute and visualize acceleration
    ComputeAndVisualizeAcceleration.compute(experiment_name)
    ComputeAndVisualizeAcceleration.visualize_verifier(
        experiment_name, show_X=True, show_Y=True, show_Z=True, PLOT=PLOT)
    ComputeAndVisualizeAcceleration.visualize_sender(
        experiment_name, show_X=True, show_Y=True, show_Z=True, PLOT=PLOT)

    # compute similarity
    legit_sim = ComputeSimilarity.compute_simularity(experiment_name)
    attack_sim = ComputeSimilarity.compute_simularity(
        experiment_name, is_attacker=True)

    # print result
    if LOG:
        print("Legitimate similarity score (over 100): {}".format(legit_sim))
        print("Attacker similarity score (over 100): {}".format(attack_sim))

    global TP, FP
    # classify
    if legit_sim > T:
        TP = TP + 1
    if attack_sim > T:
        FP = FP + 1


def compute_TPR_FPR(exp_list):
    for threshold in range(0, 101, 5):
        global TP, FP
        TP = 0
        FP = 0
        counter = 0
        for i in exp_list:
            counter = counter + 1
            #print("Doing experiment {}".format(i))
            experiment_name = "exp{}".format(i)
            do_experiment(experiment_name, threshold)
        TPR = round(TP / counter, 4)
        FPR = round(FP / counter, 4)
        print("TPR = {}, FPR = {}, Threshold = {}".format(TPR, FPR, threshold))


if __name__ == "__main__":
    exp_list = [3, 4, 5, 6, 10, 11, 12, 21, 22, 23]
    # extract data first, if needed
    # for i in exp_list:
    #    experiment_name = "exp{}".format(i)
    #    extract_data(experiment_name)

    # conduct all experiments to compute TPR, FPR based on various thresholds
    compute_TPR_FPR(exp_list)
