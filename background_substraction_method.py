import cv2
import numpy as np
from utils import frames_to_video

#pobranie video z pliku
#cap = cv2.VideoCapture('C:/Users/zocha/OneDrive - Politechnika Warszawska/6sem/wma/projekt/dane_filmy/mp4/17_08_00.mp4')
cap = cv2.VideoCapture('C:/Users/zocha/OneDrive - Politechnika Warszawska/6sem/wma/projekt/dane_filmy/mp4/renault_wjazd_dzien_dach.mp4')

frames = []

#sprawdzenie czy video zostało poprawnie wczytane
if (cap.isOpened() == False): 
  print("Error opening video stream or file")

#backSub = cv2.createBackgroundSubtractorMOG2()
backSub = cv2.createBackgroundSubtractorKNN()

#pętla główna - iteracja po wszystkich klatkach filmu
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    fgMask = backSub.apply(frame)
    kernel = np.ones((3,3), np.uint8)
    fgMask = cv2.erode(fgMask, kernel, iterations=1)
    fgMask = cv2.dilate(fgMask, kernel, iterations=1)

    kernel2 = np.ones((4,1), np.uint8)
    fgMask = cv2.erode(fgMask, kernel2, iterations=2)
    fgMask = cv2.dilate(fgMask, kernel2, iterations=3)
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    # canny_edge = cv2.Canny(fgMask, 190, 200)
    # cv2.imshow('edge', canny_edge)

    contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0,0, 255), 1)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    cv2.imshow("frame", frame)
    frames.append(frame)
    cv2.imshow('FG Mask', fgMask)
 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


frames_to_video(frames, 'bs.avi')
cap.release()
cv2.destroyAllWindows()
