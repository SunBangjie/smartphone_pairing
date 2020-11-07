import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import time
import math

DEBUG = True

FACTOR = 2
RESO_X = int(576 / FACTOR)
RESO_Y = int(640 / FACTOR)

CONF_VAL = 0
THRESHOLD = 0

UPPER_BOUND = 230
LOWER_BOUND = 150


def get_file_index(filename):
    index = int(filename.split('.')[0])
    return index


def create_windows():
    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB", RESO_X, RESO_Y)
    cv2.resizeWindow("Depth", RESO_X, RESO_Y)


def load_yolo(model_folder):
    # load the COCO class labels our YOLO model was trained on
    labelsPath = model_folder + "coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    weightsPath = model_folder + "yolov3-spp.weights"
    configPath = model_folder + "yolov3-spp.cfg"
    print("[INFO] loading YOLO from disk...")
    if DEBUG:
        print("label: {}\nweights: {}\nconfig: {}".format(
            labelsPath, weightsPath, configPath))
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln, LABELS


def process_frame(frame, net, ln, LABELS):
    # get frame height and width
    (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start_time = time.time()
    layerOutputs = net.forward(ln)
    duration = time.time() - start_time
    if DEBUG:
        print("[INFO] processed within {}s".format(round(duration, 2)))

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONF_VAL and LABELS[classID] == "cell phone":
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates and confidences
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
    return boxes, confidences


def main(rgb_folder, depth_folder, model_folder, output_folder, save_images=False):
    # load rgb images
    print("[INFO] loading rgb images from disk...")
    img_files = [f for f in listdir(rgb_folder) if isfile(join(rgb_folder, f))]
    img_files = sorted(img_files, key=get_file_index)

    # load image net
    net, ln, LABELS = load_yolo(model_folder)

    out_file = open(output_folder + "/" + "positions.txt", "w")

    # process each frame
    for img_file in img_files:

        if DEBUG:
            print("[INFO] processing image {}".format(img_file))
        # read rgb frame
        frame = cv2.imread(rgb_folder + "/" + img_file, cv2.IMREAD_COLOR)

        # read depth frame
        depth = cv2.imread(depth_folder + "/" + img_file)

        # rotate 90 degree for phone images
        # frame = cv.rotate(frame, rotateCode=cv.ROTATE_90_CLOCKWISE)

        # process using YOLO
        boxes, confidences = process_frame(frame, net, ln, LABELS)
        # suppress boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_VAL, THRESHOLD)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # get first box
            i = idxs.flatten()[0]
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(depth, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if save_images:
                # display and save image
                cv2.imshow("RGB", frame)
                cv2.imwrite(output_folder +
                            "rgb/" + img_file, frame)
                cv2.imshow("Depth", depth)
                cv2.imwrite(output_folder +
                            "depth/" + img_file, depth)

            # get centroid of the bouding box
            centroid_x = x + int(w / 2)
            centroid_y = y + int(h / 2)
            # get average depth within the bounding box
            depth_pixels = depth[x: x+w, y: y+h, 0]
            depth_pixels = depth_pixels.flatten()
            mask = (depth_pixels > LOWER_BOUND) & (depth_pixels < UPPER_BOUND)
            depth_pixels = depth_pixels[mask]
            pixel_mean = np.mean(depth_pixels)

            # save timestamp and position
            if not math.isnan(pixel_mean):
                timestamp = img_file.split('.')[0]
                out_file.write("{},{},{},{}\n".format(
                    timestamp, centroid_x, centroid_y, round(pixel_mean, 4)
                ))

            if DEBUG:
                print("point is ({}, {}, {})".format(
                    centroid_x, centroid_y, round(pixel_mean, 4)))

        key = cv2.waitKey(50)
        if key != -1:
            cv2.destroyAllWindows()
            break

    out_file.close()


if __name__ == "__main__":
    for i in [10, 11, 12]:
        experiment_name = "exp{}".format(i)
        print("Doing experiment {}".format(i))
        rgb_folder = "Experiment_Frames/" + experiment_name + "/rgb_frames/"
        depth_folder = "Experiment_Frames/" + experiment_name + "/depth_frames/"
        model_folder = "yolo-coco/"
        output_folder = "Experiment_Output/" + experiment_name + "/"
        start = time.time()
        print("[INFO] start processing all frames...")
        main(rgb_folder, depth_folder, model_folder,
            output_folder, save_images=True)
        elapse = time.time() - start
        print("[INFO] completed in {}s".format(round(elapse, 1)))
