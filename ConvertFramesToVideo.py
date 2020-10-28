import cv2
import numpy as np
import os

from os.path import isfile, join


def convert_frames_to_video(pathIn, pathOut, fps, label):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the file names properly
    files.sort(key=lambda x: int(x.split('.')[0]))

    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        # inserting the frames into an image array
        if label in filename:
            frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def convert(label):
    pathIn = './output/labelled_images/' + label + "/"
    pathOut = './output/videos/' + label + '.avi'
    fps = 30.0
    convert_frames_to_video(pathIn, pathOut, fps, label)


if __name__ == "__main__":
    # Make sure have some figs in saved_figs/
    # Make sure saved_videos/ is created
    convert('rgb')
    convert('depth')
