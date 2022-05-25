import cv2
import sys
from utils import frames_to_video
from datetime import datetime
from frame_processing import background_substraction, draw_box_contours, adaptive_threshold

PATH = "C:/Users/zocha/OneDrive - Politechnika Warszawska/6sem/wma/projekt/dane_filmy/mp4/17_08_00.mp4"
METHOD_NAME = "background_substraction"

# file export utils
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
fname = f"./output/{METHOD_NAME}_{date_time}.avi"
frames = []


try:
    f = open(PATH)
    cap = cv2.VideoCapture(PATH)
except FileNotFoundError:
    print("File does not exist")
    sys.exit(1)


background = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if background is None:
        background = frame
        continue

    if METHOD_NAME == "background_substraction":
        processed = background_substraction(frame)
    elif METHOD_NAME == "adaptive_threshold":
        processed = adaptive_threshold(frame, background)

    elif METHOD_NAME == "dnn_yolo":
        processed = frame  # TODO
    else:
        processed = frame

    frame_with_detection = draw_box_contours(processed, frame)
    cv2.imshow("frame", frame_with_detection)
    cv2.imshow("processed", processed)
    frames.append(frame_with_detection)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


frames_to_video(frames, fname)
cap.release()
cv2.destroyAllWindows()
