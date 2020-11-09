import VerifierCameraReader
import PhoneDetector
import VisualizePositions
import ComputeAndVisualizeAcceleration
import ComputeSimilarity
import Utils.ConvertFramesToVideo

import time


def do_experiment(experiment_name):
    # extract frames
    # VerifierCameraReader.extract_frames(experiment_name)

    # extract RGBD data from frames
    # PhoneDetector.detect(experiment_name)

    # visualize positions
    # VisualizePositions.visualize_position(experiment_name)
    '''
    # compute and visualize acceleration
    ComputeAndVisualizeAcceleration.compute(experiment_name)
    ComputeAndVisualizeAcceleration.visualize_verifier(
        experiment_name, show_X=True, show_Y=True, show_Z=True)
    ComputeAndVisualizeAcceleration.visualize_sender(
        experiment_name, show_X=True, show_Y=True, show_Z=True)
    '''

    # compute similarity
    ComputeSimilarity.compute_simularity(experiment_name)
    ComputeSimilarity.compute_simularity(experiment_name, is_attacker=True)


if __name__ == "__main__":
    for i in [21, 22, 23]:
        print("Doing experiment {}".format(i))
        experiment_name = "exp{}".format(i)
        do_experiment(experiment_name)
