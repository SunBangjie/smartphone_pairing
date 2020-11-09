#!/usr/bin/env python3

import h5py
import sys
import time
import cv2
import numpy as np
import os


# Call as `python kinect_read.py <filename>`


def extract_frames(experiment_name):
    # For user-resizable
    cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("RGB", cv2.WINDOW_AUTOSIZE)

    filename = "./Experiment_Data/{}/{}.hdf5".format(
        experiment_name, experiment_name)
    if not os.path.exists('./Experiment_Frames/{}/depth_frames'.format(experiment_name)):
        os.makedirs(
            './Experiment_Frames/{}/depth_frames'.format(experiment_name))
    if not os.path.exists('./Experiment_Frames/{}/rgb_frames'.format(experiment_name)):
        os.makedirs('./Experiment_Frames/{}/rgb_frames'.format(experiment_name))
    with h5py.File(filename, "r") as f:
        timestamps = f["timestamp"]
        depth = f["depth"]
        color = f["color"]
        for i in range(depth.shape[0]):
            timestamp = str(timestamps[i][0])
            depth_img = depth[i]
            depth_img = cv2.normalize(
                depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow("Depth", depth_img)
            cv2.imwrite(
                "./Experiment_Frames/{}/depth_frames/{}.jpg".format(experiment_name, timestamp), depth_img)
            cv2.imshow("RGB", color[i])
            cv2.imwrite(
                "./Experiment_Frames/{}/rgb_frames/{}.jpg".format(experiment_name, timestamp), color[i])

            # Simulate 50ish fps
            key = cv2.waitKey(20)
            if key != -1:
                cv2.destroyAllWindows()
                break
