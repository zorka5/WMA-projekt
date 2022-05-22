import cv2
import numpy as np

#pobranie video z pliku
cap = cv2.VideoCapture('./data/17_08_00.mp4')
print("cap type", type(cap))

#array to store processed frames to crate output video
img_array = []

whT = 320

#constants of dnn
confidenceThreshold = 0.50
nmsThreshold = 0.3


#creating dnn model
classesFile = './dnn_files/coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


modelConfiguration = './dnn_files/yolov3.cfg'
modelWeights = './dnn_files/yolov3.weights'
model = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


def findObjects(outputs, img):
    height, width, center = img.shape
    boundingBox = []
    classIds = []
    confidence_values = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence: float = scores[classId]
            if confidence > confidenceThreshold:
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(detection[0]*width - w/2)
                y = int(detection[1]*height - h/2)
                boundingBox.append([x,y,w,h])
                classIds.append(classId)
                confidence_values.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(boundingBox, confidence_values, confidenceThreshold, nmsThreshold)
    for i in indices:
        box = boundingBox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255))
        cv2.putText(img, f'{classNames[classIds[i]].upper()}: {int(confidence_values[i]*100)}%', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

#sprawdzenie czy video zostało poprawnie wczytane
if (cap.isOpened() == False): 
  print("Error opening video stream or file")

#backSub = cv2.createBackgroundSubtractorMOG2()
backSub = cv2.createBackgroundSubtractorKNN()

size = (0,0)

#pętla główna - iteracja po wszystkich klatkach filmu
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break 

    blob = cv2.dnn.blobFromImage(frame, 1/255, (whT, whT), [0,0,0], 1, crop=False)
    model.setInput(blob)

    layerNames = model.getLayerNames()
    outputNames = [layerNames[i-1] for i in model.getUnconnectedOutLayers()]
    outputs = model.forward(outputNames)
    findObjects(outputs, frame)
    

    cv2.imshow("frame", frame)

    img_array.append(frame)
    h,w,layers = frame.shape
    size = (w,h)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("done")
#out = cv2.VideoWriter('./data/yolo.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size=size)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('yolo.avi', fourcc, 20, size)
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()

print("writing done")
print("type of img_array", type(img_array))


cap.release()
cv2.destroyAllWindows()