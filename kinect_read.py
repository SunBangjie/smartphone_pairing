#!/usr/bin/env python3

import h5py
import sys
import time
from helpers import colorize
import cv2
import numpy as np


# Call as `python kinect_read.py <filename>`


def main(filename):
    # For user-resizable
    cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Color", cv2.WINDOW_AUTOSIZE)

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
            cv2.imwrite("./depth_frames/{}.jpg".format(timestamp), depth_img)
            cv2.imshow("Color", color[i])
            cv2.imwrite("./rgb_frames/{}.jpg".format(timestamp), color[i])

            # Simulate 30ish fps
            key = cv2.waitKey(33)
            if key != -1:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    main(sys.argv[1])
