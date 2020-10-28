import cv2
import numpy as np

import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A
from datetime import datetime

# Store into HDF5 dataset
# https://docs.h5py.org/en/stable/high/dataset.html
import h5py

import sys

# Profiling
import cProfile
import time


# For concating: https://stackoverflow.com/questions/25655588/incremental-writes-to-hdf5-with-h5py?rq=1

# Size of each element with current Config settings
# Color size: (720, 1280, 4)
# Transformed Depth size : (720, 1280)
# IR size: (576, 640)

RGBD_X = 576
RGBD_Y = 640
COLOR_Z = 4

# How many frames should we keep in RAM before writing to disk?
RGBD_WRITE_BUFFER_NUM_FRAMES = 1

# Create a HDF5 file with current timestamp and prints the filename to screen


def create_timestamped_hdf5_file():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = current_time + ".hdf5"
    f = h5py.File(filename, "w")
    print(f"Session file name: {filename}")
    return f


def create_datasets_and_return(f: h5py.File):
    # https://docs.h5py.org/en/stable/high/dataset.html
    # RGB: 720 x 1280 x 4 color channels x num images
    timestampset = f.create_dataset(
        name="timestamp",
        shape=(1, 1),
        maxshape=(None, 1),
        dtype="int64"
    )

    colorset = f.create_dataset(
        name="color",
        shape=(1, RGBD_X, RGBD_Y, COLOR_Z),
        maxshape=(None, RGBD_X, RGBD_Y, COLOR_Z),
        compression="gzip",
        # Set this accordingly to trade off FPS and filesize
        compression_opts=0,
        dtype="uint8",
    )

    depthset = f.create_dataset(
        name="depth",
        shape=(1, RGBD_X, RGBD_Y),
        maxshape=(None, RGBD_X, RGBD_Y),
        compression="gzip",
        # compression_opts=9,
        dtype="uint16",
    )

    return colorset, depthset, timestampset


def create_buffers_and_return():
    timstampbuffer = np.zeros(
        shape=(RGBD_WRITE_BUFFER_NUM_FRAMES, 1),
        dtype="int64"
    )
    colorbuffer = np.zeros(
        shape=(RGBD_WRITE_BUFFER_NUM_FRAMES, RGBD_X, RGBD_Y, COLOR_Z),
        dtype="uint16",
    )
    depthbuffer = np.zeros(
        shape=(RGBD_WRITE_BUFFER_NUM_FRAMES, RGBD_X, RGBD_Y),
        dtype="uint16",
    )
    return colorbuffer, depthbuffer, timstampbuffer


def resize_dataset_and_write(dataset, data):
    """
    Assumes we are resizing dataset.shape[0] by adding data.shape[0] elements to it
    """
    num_new_elements = len(data)
    dataset.resize(dataset.shape[0] + num_new_elements, axis=0)
    dataset[-num_new_elements:] = data


def main():
    # Open storage file
    f = create_timestamped_hdf5_file()

    # Create datasets for each stream
    colorset, depthset, timestampset = create_datasets_and_return(f)
    # colorbuffer, irbuffer, depthbuffer = create_buffers_and_return()
    colorbuffer, depthbuffer, timestampbuffer = [], [], []

    # Start session to the kinect
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # Use "window" as first arg to get fullscreen
    # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Enable user-resizable windows
    cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Color", cv2.WINDOW_AUTOSIZE)

    start_time = time.time()
    x = 1  # displays the frame rate every 1 second
    counter = 0
    while True:
        capture = k4a.get_capture()
        if (
            capture.depth is not None
            and capture.transformed_color is not None
        ):

            # Show depth and color images
            cv2.imshow("Depth", colorize(
                capture.depth, (None, None)))
            cv2.imshow("Color", capture.transformed_color)

            # Buffered writes to the file, but expts show not buffering is better
            timestampbuffer.append(
                int(datetime.timestamp(datetime.now()) * 1000))
            colorbuffer.append(capture.transformed_color)
            depthbuffer.append(capture.depth)
            if len(colorbuffer) == RGBD_WRITE_BUFFER_NUM_FRAMES:
                resize_dataset_and_write(timestampset, timestampbuffer)
                resize_dataset_and_write(colorset, colorbuffer)
                resize_dataset_and_write(depthset, depthbuffer)
                timestampbuffer.clear()
                depthbuffer.clear()
                colorbuffer.clear()

            # Actually WRITE to to h5 file
            # colorset[-1:] = capture.transformed_color
            # depthset[-1:] = capture.depth

            # FPS calc
            counter += 1
            if (time.time() - start_time) > x:
                print("FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()

            # We need this or the frame won't redraw
            key = cv2.waitKey(1)
            if key != -1:
                cv2.destroyAllWindows()
                break

    # Stop session after loop breaks
    k4a.stop()
    f.close()


PROFILE = False
if __name__ == "__main__":
    if PROFILE:
        import re

        cProfile.run("main()", sort="cumulative")
    else:
        main()
