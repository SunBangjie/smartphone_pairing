import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join


IMAGE_THRESHOLD_LOWER_BOUND = 200

SHAPE_SCORE = 500
AREA_SCORE = 1
ARC_SCORE = 0.5

FACTOR = 2
RESO_X = int(576 / FACTOR)
RESO_Y = int(640 / FACTOR)


def create_windows():
    cv.namedWindow("Original Image", cv.WINDOW_NORMAL)
    cv.namedWindow("Pre-processed Image", cv.WINDOW_NORMAL)
    cv.namedWindow("Edges", cv.WINDOW_NORMAL)
    cv.namedWindow("Selected Contour", cv.WINDOW_NORMAL)
    cv.namedWindow("Fitted Rectangle", cv.WINDOW_NORMAL)
    cv.namedWindow("Extreme Points", cv.WINDOW_NORMAL)
    cv.namedWindow("Turning Points", cv.WINDOW_NORMAL)
    cv.resizeWindow("Original Image", RESO_X, RESO_Y)
    cv.resizeWindow("Pre-processed Image", RESO_X, RESO_Y)
    cv.resizeWindow("Edges", RESO_X, RESO_Y)
    cv.resizeWindow("Selected Contour", RESO_X, RESO_Y)
    cv.resizeWindow("Fitted Rectangle", RESO_X * 2, RESO_Y * 2)
    cv.resizeWindow("Extreme Points", RESO_X, RESO_Y)
    cv.resizeWindow("Turning Points", RESO_X, RESO_Y)


def get_angle(P1, P2, P3):
    P1 = np.array(P1)
    P2 = np.array(P2)
    P3 = np.array(P3)
    P2P1 = P1 - P2
    P2P3 = P3 - P2
    unit_P2P1 = P2P1 / np.linalg.norm(P2P1)
    unit_P2P3 = P2P3 / np.linalg.norm(P2P3)
    dot_product = np.dot(unit_P2P1, unit_P2P3)
    angle = np.arccos(dot_product)
    return angle * 180 / 3.1415926


def get_file_index(filename):
    index = int(filename.split('.')[0])
    return index


def get_contour_score(contour):
    overall_score = 0
    area = cv.contourArea(contour)
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.04 * peri, True)
    if len(approx) > 3:
        overall_score = overall_score + SHAPE_SCORE
    overall_score = overall_score + area * AREA_SCORE + peri * ARC_SCORE
    return overall_score


def pre_process(img):
    # Smoothing
    img = cv.GaussianBlur(img, (3, 3), 0)
    img = cv.medianBlur(img, 3)

    # Filtering
    ret, thresh = cv.threshold(
        img, IMAGE_THRESHOLD_LOWER_BOUND, 255, cv.THRESH_BINARY)
    return thresh


def detect_edges(img):
    edges = cv.Canny(img, 127, 255)
    return edges


def detect_contours(img):
    ret, thresh = cv.threshold(img, 127, 255, 0)
    im2, contours, hierarchy = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if len(contours) < 0:
        raise Exception
    # sort contours from smallest area to largest
    sorted_contours = sorted(contours, key=get_contour_score)
    largest_contour = sorted_contours[-1]
    # generate fitted box
    fitted_rect = cv.minAreaRect(largest_contour)
    fitted_box = cv.boxPoints(fitted_rect)
    fitted_box = np.int0(fitted_box)
    # generate centroid
    M = cv.moments(largest_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid = (cX, cY)
    c = largest_contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    extreme_points = [extLeft, extRight, extTop, extBot]
    # find turning angle
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.04 * peri, True)
    turning_points = []
    if len(approx) >= 3:
        for i in range(len(approx)):
            angle = get_angle(approx[i-1][0], approx[i]
                              [0], approx[(i+1) % len(approx)][0])
            if 80 < angle < 100:
                tp_tuple = (approx[i][0][0], approx[i][0][1])
                turning_points.append(tp_tuple)
    # return largest contour, fitted box and centroid of contour
    return largest_contour, fitted_box, centroid, extreme_points, turning_points


def detect_corners(img):
    # Use Canny detector
    dst = cv.cornerHarris(img, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS +
                cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(img, np.float32(
        centroids), (5, 5), (-1, -1), criteria)
    # now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    return res


def detect(folder):
    img_files = [f for f in listdir(folder) if isfile(join(folder, f))]
    img_files = sorted(img_files, key=get_file_index)

    # create windows for display
    create_windows()

    for img_file in img_files:
        # Read depth images
        img = cv.imread(folder + "/" + img_file, 0)
        # crop image
        img = img[50:500, 95:545]
        # img = cv.rotate(img, rotateCode=cv.ROTATE_90_CLOCKWISE)
        cv.imshow("Original Image", img)

        # Pre-process image by blurring
        processed_img = pre_process(img)
        cv.imshow("Pre-processed Image", processed_img)

        # Use Canny edge detector
        edges = detect_edges(processed_img)
        cv.imshow("Edges", edges)

        # Use Harris corner detector
        # mask = detect_corners(edges)
        # gray = np.zeros(edges.shape)
        # gray[mask[:, 1], mask[:, 0]] = 0  # centroids (black)
        # gray[mask[:, 3], mask[:, 2]] = 255  # corners (white)
        #cv.imshow("Corners", gray)

        # Detect contours and fit into shapes
        try:
            contour, box, centroid, extreme_points, turning_points = detect_contours(
                edges)
        except:
            continue
        contour_img = np.copy(img)
        cv.drawContours(contour_img, [contour], 0, 255, 2)
        cv.imshow("Selected Contour", contour_img)
        box_img = np.copy(img)
        cv.drawContours(box_img, [box], 0, 255, 2)
        cv.imshow("Fitted Rectangle", box_img)
        points_img = np.copy(img)
        cv.circle(points_img, centroid, 10, 255, -1)
        for ep in extreme_points:
            cv.circle(points_img, ep, 5, 255, -1)
        cv.imshow("Extreme Points", points_img)
        turning_img = np.copy(img)
        for tp in turning_points:
            cv.circle(turning_img, tp, 5, 255, -1)
        cv.imshow("Turning Points", turning_img)

        key = cv.waitKey(50)
        if key != -1:
            cv.destroyAllWindows()
            break


if __name__ == "__main__":
    detect("./Experiment_Frames/exp15/rgb_frames")
