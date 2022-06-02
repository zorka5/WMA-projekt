import cv2
import sys
import numpy as np
from utils import frames_to_video
from mask_creation import create_mask
from datetime import datetime
from frame_processing import (
    background_substraction,
    draw_box_contours,
    adaptive_threshold,
    mask_color,
    color_segmentation,
)

# PATH = "C:/Users/zocha/OneDrive - Politechnika Warszawska/6sem/wma/projekt/dane_filmy/mp4/renault_wjazd_dzien_dach.mp4"
PATH = "C:/Users/zocha/OneDrive - Politechnika Warszawska/6sem/wma/projekt/dane_filmy/mp4/17_08_00.mp4"
METHOD_NAME = "color_segmentation"

# file export utils
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
fname_processed = f"./output/{METHOD_NAME}_{date_time}_detection.avi"
fname_detection = f"./output/{METHOD_NAME}_{date_time}_processed.avi"
processed_frames = []
frames_with_detection = []


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

    processed = frame

    # maska wyczerniająca otoczenie
    mask = create_mask("./data/background_mask.jpg")
    processed = cv2.bitwise_and(frame, frame, mask=mask)

    # stworzenie maski bramy

    # odejmowanie tła
    processed = background_substraction(processed)

    # progowanie w celu wyeliminowania szarego koloru
    ret, processed = cv2.threshold(processed, 2, 255, cv2.THRESH_BINARY)

    # zaznaczanie konturów i obrysowywanie wykrytego ruchu w prostokątach
    frame_with_detection = draw_box_contours(processed, frame)

    # wyświetlanie i zapis klatek do tablicy
    cv2.imshow("frame", frame_with_detection)
    cv2.imshow("processed", processed)
    frames_with_detection.append(frame_with_detection)
    processed_frames.append(processed)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# frames_to_video(frames_with_detection, fname_detection)
# frames_to_video(processed_frames, fname_processed)
cap.release()
cv2.destroyAllWindows()
