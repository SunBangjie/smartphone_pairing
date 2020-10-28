# Smartphone Pairing
Here is the list of things done:
1. An Android app (in ./app/) to collect accelerometer data on the sender.
2. A TCP server in Python (./Server.py) to receive the accelerometer data and write into ./acc_reading.txt
3. A script (./EdgeCornerDetector.py) to analyze each depth frame to detect and track a smartphone.
4. A script (./PerFrameObjectDetector.py) to detect cell phones in a video frame using YOLO v3 DNN.
5. Scripts to compute frame velocity (./ComputeFrameVelocity.py) and visualize in 3 axes (./VisualizeVelocity.py).
6. Other utility scripts (kinect_read, kinect_write, ConvertFramesToVideo, ConvertVideoToFrames)
