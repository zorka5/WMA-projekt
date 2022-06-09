import numpy as np
import cv2


whT = 320


def frames_to_video(frames_array: list, filename: str) -> None:
    h = frames_array[0].shape[0]
    w = frames_array[0].shape[1]
    size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(filename, fourcc, 20, size)
    for i in range(len(frames_array)):
        out.write(frames_array[i])
    out.release()


# pobranie video z pliku
filename = "toyota_dzien_wjazd"
capture_path = f"D:/Documents/Projects/WMA-projekt/data/{filename}.mp4"

threshold = 0.7
nmsThreshold = 0.1

img_array2 = []


class CarDetection:
    def __init__(self, cap_path_, classFile, filename):
        self.cap_path = cap_path_
        self.cap = cv2.VideoCapture(f"./data/{filename}.mp4")
        self.modelConfiguration = "./dnn_files/yolov3.cfg"
        self.modelWeights = "./dnn_files/yolov3.weights"
        self.model = cv2.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        self.classNames = []
        with open(classFile, "rt") as f:
            self.classNames = f.read().rstrip("\n").split("\n")
        self.img_array = []
        self.size = (0, 0)

    def find_objects(self, outputs, img):
        height, width, center = img.shape
        boundingBox = []
        classIds = []
        confidence_values = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence: float = scores[classId]
                if confidence > threshold:
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(detection[0] * width - w / 2)
                    y = int(detection[1] * height - h / 2)
                    boundingBox.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_values.append(float(confidence))
        indices = cv2.dnn.NMSBoxes(boundingBox, confidence_values, threshold, nmsThreshold)
        for i in indices:
            box = boundingBox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))
            cv2.putText(
                img,
                f"{self.classNames[classIds[i]].upper()}: {int(confidence_values[i]*100)}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    def f(self):
        if self.cap.isOpened() is False:
            print("Error opening video stream or file")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
            self.model.setInput(blob)

            layerNames = self.model.getLayerNames()
            outputNames = [layerNames[i - 1] for i in self.model.getUnconnectedOutLayers()]
            outputs = self.model.forward(outputNames)
            self.find_objects(outputs, frame)
            cv2.imshow("frame", frame)

            self.img_array.append(frame)
            # h, w, layers = frame.shape
            # size = (w, h)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


# Create a new object and execute.
detector = CarDetection(cap_path_=capture_path, classFile="./dnn_files/coco.names", filename="toyota_dzien_wjazd")
detector.f()


print(len(detector.img_array))
print(type(detector.img_array))
print(detector.img_array)

print("done")
frames_to_video(detector.img_array, f"./output/yolov3_{filename}.avi")
print("writing done")
