import cv2


def frames_to_video(frames_array: list, filename: str) -> None:
    h = frames_array[0].shape[0]
    w = frames_array[0].shape[1]
    size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(filename, fourcc, 20, size)
    for i in range(len(frames_array)):
        out.write(frames_array[i])
    out.release()
