import cv2
import numpy as np

#load Yolo
net = cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg.txt")
classes = []

with open("coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#load image
img = cv2.imread("birds2.jpg")
H, W = (img.shape[0], img.shape[1])

#for image resize
frame = cv2.resize(img, None, fx=0.4, fy=0.4)


H, W = (frame.shape[0], frame.shape[1])
height, width, channels = frame.shape

blob =cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)
output = net.forward(output_layers)

boxes = []
confidences = []
classIDs = []

for out in output:
    for detection in out:

        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        #classId == 14 means that it only reads detection of birds based on coco.names.txt
        if confidence > 0.5 and classID == 14:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype('int')

            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            box_centers = [centerX, centerY]

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    for i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[classIDs[i]])
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)

cv2.imshow("Image",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
