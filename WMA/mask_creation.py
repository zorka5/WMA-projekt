import cv2
import numpy as np


def create_mask(filename: str):
    background_mask = cv2.imread(filename)
    gray = cv2.cvtColor(background_mask, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)
    ret, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    mask = cv2.inRange(th2, 0, 1)
    mask = cv2.bitwise_not(mask)
    return mask


cv2.waitKey(0)
