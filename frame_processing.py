import cv2
import numpy as np

backSub = cv2.createBackgroundSubtractorKNN()


def background_substraction(frame: np.ndarray) -> np.ndarray:
    fgMask = backSub.apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    fgMask = cv2.erode(fgMask, kernel, iterations=1)
    fgMask = cv2.dilate(fgMask, kernel, iterations=1)

    kernel2 = np.ones((4, 1), np.uint8)
    fgMask = cv2.erode(fgMask, kernel2, iterations=2)
    fgMask = cv2.dilate(fgMask, kernel2, iterations=3)

    return fgMask


def adaptive_threshold(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    # blank = np.zeros(frame.shape[:3])

    diff = cv2.absdiff(background, frame)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = diff.astype(np.uint8)

    th2 = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 5)
    # cv2.imshow("diff", th2)
    # th2 = cv2.GaussianBlur(th2,(3,3),cv2.BORDER_DEFAULT)
    kernel = np.ones((2, 2), np.uint8)
    th2 = cv2.erode(th2, kernel, iterations=2)
    th2 = cv2.dilate(th2, kernel, iterations=1)
    # cv2.imshow("diff thresholded", th2)

    return th2


def draw_box_contours(input_frame: np.ndarray) -> np.ndarray:
    contours, hierarchy = cv2.findContours(input_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    blank = np.zeros(input_frame.shape[:3])
    output_frame = cv2.drawContours(blank, contours, -1, (0, 0, 255), 1)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return output_frame
