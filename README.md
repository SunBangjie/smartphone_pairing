# Smartphone Pairing
Here is the list of things done:
1. An Android app (in ./app/) to collect accelerometer data on the sender.
2. A TCP server in Python (./Server.py) to receive the accelerometer data and write into ./acc_reading.txt
3. A script (./ConvertToFrames.py) to convert ./Depth_Video.mp4 into a sequence of frames into ./depth_frames/
4. A script (./EdgeCornerDetector.py) to analyze each depth frame to detect and track a smartphone.
