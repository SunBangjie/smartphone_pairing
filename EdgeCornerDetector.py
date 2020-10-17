import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join


def get_file_index(filename):
    name = filename.split('.')[0]
    index = int(name.split('_')[1])
    return index


def pre_process(img):
    img = cv.GaussianBlur(img, (5, 5), 0)
    img = cv.medianBlur(img, 5)
    return img


def detect_edges(img):
    edges = cv.Laplacian(img, cv.CV_8U, ksize=5)
    ret, mask = cv.threshold(edges, 100, 255, cv.THRESH_BINARY_INV)
    return mask


def detect_corners(edges):
    # Use Canny detector
    dst = cv.cornerHarris(edges, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS +
                cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(edges, np.float32(
        centroids), (5, 5), (-1, -1), criteria)
    # now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    return res


def detect(folder):
    depth_files = [f for f in listdir(folder) if isfile(join(folder, f))]
    depth_files = sorted(depth_files, key=get_file_index)

    cv.namedWindow("Original Depth Image", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Pre-processed Image", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Edges", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Corners", cv.WINDOW_AUTOSIZE)

    for depth_image in depth_files:
        # Read depth images
        img = cv.imread(folder + "/" + depth_image, 0)
        cv.imshow("Original Depth Image", img)

        # Pre-process image by blurring
        img = pre_process(img)
        cv.imshow("Pre-processed Image", img)

        # Use Canny edge detector
        edges = detect_edges(img)
        cv.imshow("Edges", edges)

        # Use Harris corner detector
        mask = detect_corners(edges)
        gray = np.ones(img.shape)
        gray[mask[:, 1], mask[:, 0]] = 255  # centroids
        gray[mask[:, 3], mask[:, 2]] = 0  # corners
        cv.imshow("Corners", gray)

        key = cv.waitKey(50)  # 1000/50 = 20 FPS
        if key != -1:
            cv.destroyAllWindows()
            break


if __name__ == "__main__":
    detect("./depth_frames")
