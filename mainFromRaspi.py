import cv2
import numpy as np
import time
import RPi.GPIO as GPIO  
from time import sleep
from playsound import playsound

#load Yolo
net = cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg.txt")
classes = []
with open("coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

in1 = 24
in2 = 23

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)

def run():
    playsound('audio.wav')
    in1 = 24
    in2 = 23
    print("Bird Detected")
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    sleep(1)
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)

exchange_camera = True
first_load = True
using_zero = True
on_repeat = True

while on_repeat:
    print("Change Camera")
    time_default = time.time()
    if first_load:
        cap = cv2.VideoCapture(0)
    elif not first_load:
        if using_zero:
            cap = cv2.VideoCapture(1)
            using_zero = False
        else:
            cap = cv2.VideoCapture(0)
            using_zero = True

    first_load = False
    while cap.isOpened():
        success, frame = cap.read()
        time_start = time.time()

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
                if confidence > 0.3 and classID==14:
                    #insert function para sa motor
                    run()

        cv2.imshow("Image", frame) #hangtod ani
        #for video
        if cv2.waitKey(5) & 0xFF == ord('q'):
            on_repeat = False
            break
        if (time_start-time_default > 25):
            cv2.destroyAllWindows()
            break
