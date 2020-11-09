import numpy as np
import cv2 as cv
from os import listdir, makedirs
from os.path import isfile, join, exists
import math
import Threshold


def get_file_index(filename):
    index = int(filename.split('.')[0])
    return index


SHAPE_SCORE = 500
AREA_SCORE = 1
ARC_SCORE = 0.5


def get_contour_score(contour):
    overall_score = 0
    area = cv.contourArea(contour)
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.04 * peri, True)
    if len(approx) > 3:
        overall_score = overall_score + SHAPE_SCORE
    overall_score = overall_score + area * AREA_SCORE + peri * ARC_SCORE
    return overall_score


def detect_contours(img):
    im2, contours, hierarchy = cv.findContours(
        img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if len(contours) <= 0:
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
    # return fitted box and centroid of contour
    return fitted_box, centroid


def color_mask(original_image):
    t = 30  # tolerance
    R = 179
    G = 79
    B = 66
    low_threshold = np.array([G - t, B - t, R - t])
    upp_threshold = np.array([G + t, B + t, R + t])
    mask = cv.inRange(original_image, low_threshold, upp_threshold)
    return mask


def detect(experiment_name):

    # Get inputs
    rgb_folder = "./Experiment_Frames/{}/rgb_frames/".format(experiment_name)
    depth_folder = "./Experiment_Frames/{}/depth_frames/".format(
        experiment_name)
    img_files = [f for f in listdir(
        rgb_folder) if isfile(join(rgb_folder, f))]
    img_files = sorted(img_files, key=get_file_index)

    # Set up outputs
    output_folder = "Experiment_Output/" + experiment_name + "/"
    if not exists(output_folder):
        makedirs(output_folder)

    # open outfile to write
    out_file = open(output_folder + "/" + "positions.txt", "w")

    # create windows for display
    #cv.namedWindow("Original Image", cv.WINDOW_AUTOSIZE)
    #cv.namedWindow("Pre-processed Image", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Contour Image", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Depth Image", cv.WINDOW_AUTOSIZE)

    for img_file in img_files:
        # Read rgb images
        rgb_path = rgb_folder + img_file
        depth_path = depth_folder + img_file
        img = cv.imread(rgb_path)
        #cv.imshow("Original Image", img)

        # Read depth images
        depth = cv.imread(depth_path)

        # Mask redish color
        masked_img = color_mask(img)
        #cv.imshow("Pre-processed Image", masked_img)

        try:
            # Detect contour
            fitted_box, centroid = detect_contours(masked_img)
            box_img = np.copy(img)

            # Visualize bounding box and centroid
            cv.drawContours(box_img, [fitted_box], 0, 255, 2)
            cv.circle(box_img, centroid, 1, (255, 0, 0), 1)
            cv.imshow("Contour Image", box_img)

            # Visualize mask on depth also
            depth = cv.bitwise_and(depth, depth, mask=masked_img)
            pixel_sum = np.sum(depth)
            pixel_count = np.count_nonzero(depth)
            # Only store when we have more than 1 pixel
            if pixel_count > 0:
                pixel_mean = pixel_sum / pixel_count
                # Store timestamp and position when depth is within range
                if Threshold.DEPTH_MIN <= pixel_mean <= Threshold.DEPTH_MAX:
                    timestamp = img_file.split('.')[0]
                    out_file.write("{},{},{},{}\n".format(
                        timestamp, centroid[0], centroid[1], round(
                            pixel_mean, 4)
                    ))
                print("point is ({}, {}, {})".format(
                    centroid[0], centroid[1], round(pixel_mean, 4)))
        except:
            box_img = np.copy(img)
            cv.imshow("Contour Image", box_img)

        # Show depth image
        cv.imshow("Depth Image", depth)

        key = cv.waitKey(33)
        if key != -1:
            cv.destroyAllWindows()
            break

    # close outfile
    out_file.close()


if __name__ == "__main__":
    detect("exp18")
