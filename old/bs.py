import cv2
import numpy as np
from WMA.utils import frames_to_video
from datetime import datetime

filename = "renault_wjazd_dzien_dach.mp4"
PATH = f"C:/Users/zocha/OneDrive - Politechnika Warszawska/6sem/wma/projekt/dane_filmy/mp4/{filename}"
method = "2-odejmowanie-tla"

# file export utils
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
fname_processed = f"./output/{method}_{filename}_{date_time}_detection.avi"
fname_detection = f"./output/{method}_{filename}_{date_time}_processed.avi"
processed_frames = []
frames_with_detection = []

# pobranie video z pliku
cap = cv2.VideoCapture("./data/17_08_00.mp4")

# sprawdzenie czy video zostało poprawnie wczytane
if cap.isOpened() == False:
    print("Error opening video stream or file")

# inicjalizacja tła - pierwszej klatki filmu
background = None

# tablica klatek do zapisu wideo
frames = []

backSub = cv2.createBackgroundSubtractorKNN()

# pętla główna - iteracja po wszystkich klatkach filmu
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    blank = np.zeros(frame.shape[:3])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgMask = backSub.apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    fgMask = cv2.erode(fgMask, kernel, iterations=1)
    fgMask = cv2.dilate(fgMask, kernel, iterations=1)

    kernel2 = np.ones((4, 1), np.uint8)
    fgMask = cv2.erode(fgMask, kernel2, iterations=2)
    fgMask = cv2.dilate(fgMask, kernel2, iterations=3)
    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)

    cv2.imshow("th2", fgMask)
    processed_frames.append(fgMask)

    contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    good_contours = []
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        good_contours.append(contour)
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # cv2.drawContours(frame, contours, -1, (0,255,0), 1)
    cv2.drawContours(frame, good_contours, -1, (0, 0, 255), 1)
    cv2.imshow("frame", frame)
    frames_with_detection.append(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


frames_to_video(frames_with_detection, fname_detection, 3)
frames_to_video(processed_frames, fname_processed, 0)
cap.release()
cv2.destroyAllWindows()
