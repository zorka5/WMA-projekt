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


def mask_color(
    frame: np.ndarray, lower_color_hsv: tuple[int, int, int], upper_color_hsv: tuple[int, int, int]
) -> np.ndarray:
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_frame, lower_color_hsv, upper_color_hsv)
    return mask


def color_segmentation(frame) -> np.ndarray:
    # maskowanie trwanika
    trawnik1_mask = mask_color(frame, (60, 75, 120), (75, 175, 200))
    trawnik2_mask = mask_color(frame, (70, 40, 120), (90, 120, 180))
    trawnik_mask = cv2.bitwise_or(trawnik2_mask, trawnik1_mask)

    # maskowanie drzew i krzewów
    tuje1_mask = mask_color(frame, (40, 50, 50), (100, 250, 75))
    tuje2_mask = mask_color(frame, (40, 25, 10), (80, 245, 150))
    tuje_mask = cv2.bitwise_or(tuje1_mask, tuje2_mask)

    # maskowanie chodnika

    # maskowanie kory i bramy
    kora_mask = mask_color(frame, (130, 40, 50), (175, 80, 230))
    brama_mask = mask_color(frame, (0, 0, 50), (100, 75, 100))

    brama_kora_mask = cv2.bitwise_or(brama_mask, kora_mask)
    zielone_mask = cv2.bitwise_or(trawnik_mask, tuje_mask)

    processed = cv2.bitwise_or(zielone_mask, brama_kora_mask)
    processed = cv2.bitwise_not(processed)
    fin = cv2.bitwise_and(frame, frame, mask=processed)
    return fin


def draw_box_contours(input_frame: np.ndarray, output_frame: np.ndarray) -> np.ndarray:
    contours, hierarchy = cv2.findContours(input_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.drawContours(output_frame, contours, -1, (0, 0, 255), 1)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        class_name = "aa"
        if h > w:
            class_name = "person"
        else:
            class_name = "car"
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.putText(
            output_frame,
            class_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return output
