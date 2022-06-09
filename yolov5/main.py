import torch
import numpy as np
import cv2


def frames_to_video(frames_array: list, filename: str, channels: int) -> None:
    h = frames_array[0].shape[0]
    w = frames_array[0].shape[1]
    size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(filename, fourcc, 20, size, channels)
    for i in range(len(frames_array)):
        out.write(frames_array[i])
    out.release()


# pobranie video z pliku
filename = "17_08_00"
capture_path = f"D:/Documents/Projects/WMA-projekt/data/{filename}.mp4"

threshold = 0.7

img_array = []


class CarDetection:
    def __init__(self, capture_path, model_name):
        self.capture_path = capture_path
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_video_capture(self):
        return cv2.VideoCapture(self.capture_path)

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path="D:/Documents/Projects/WMA-projekt/yolov5/best(1).pt",
                force_reload=True,
                autoshape=True,
            )  # force_reload = recache latest code
        else:
            model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            conf = row[4]
            # conf_label = float(str(conf)[7:-1]) * 100
            conf_label = round(float(conf) * 100, 2)

            print(conf_label)

            if row[4] >= threshold:
                x1, y1, x2, y2 = (
                    int(row[0] * x_shape),
                    int(row[1] * y_shape),
                    int(row[2] * x_shape),
                    int(row[3] * y_shape),
                )
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(
                    frame,
                    self.class_to_label(labels[i]) + str(conf_label) + "%",
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    bgr,
                    2,
                )

        return frame

    def __call__(self):
        cap = self.get_video_capture()
        assert cap.isOpened()

        while cap.isOpened():

            ret, frame = cap.read()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            cv2.imshow("YOLOv5 Detection", frame)
            img_array.append(frame)
            print(len(img_array))

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()


detector = CarDetection(
    capture_path=capture_path,
    model_name="best.pt",
)
detector()

print(len(img_array))
print(type(img_array))
print(img_array)

arr = np.array(img_array)
print(arr.shape)

print("done")
frames_to_video(img_array, f"./output/yolov5_{filename}.avi", 3)
print("writing done")
