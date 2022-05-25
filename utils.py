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


def get_class_names():
    classesFile = "./dnn_files/coco.names"
    classNames = []
    with open(classesFile, "rt") as f:
        classNames = f.read().rstrip("\n").split("\n")
    return classNames


def create_dnn_model():
    modelConfiguration = "./dnn_files/yolov3.cfg"
    modelWeights = "./dnn_files/yolov3.weights"
    model = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    return model
