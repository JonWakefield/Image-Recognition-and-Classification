from unittest import result
import cv2
import numpy as np
from PIL import ImageGrab
from PIL import Image, ImageFilter
import pyautogui
import datetime
import time
import os


# Object detection code
def detection():

    print("loading object detection...")

    ACCURACY = 0.60  # Threshold to detect object
    classNames = []
    classFile = 'detection\\detection_files\\coco.names'

    with open(classFile,'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
        configPath = 'detection\\detection_files\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'detection\\detection_files\\frozen_inference_graph.pb'
        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(600, 600)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

    def get_objects(img , draw=True , objects=[]):

        classIds, confs, bbox = net.detect(img, confThreshold=ACCURACY, nmsThreshold=0.1)   

        if len(objects) ==0: objects = classNames
        object_name = []
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                className = classNames[classId - 1]
                if className in objects:
                    object_name.append([className]) #object_name.append([box, className])
                    if (draw):
                        cv2.rectangle(img, box, color=(0 , 255, 0), thickness=3)
                        # cv2.putText(img, classNames[classId-1].upper(), (box[0] + 10, box[1] + 30)
                        #         cv2.FONT_HERSHEY_COMPLEX, 1, (255 , 0 , 0), 2)
                        # cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        #         cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0 , 0), 2)
        return img, str(object_name)

    # WITH VIDEO CAMERA#
    cap = cv2.VideoCapture(0) #Uses source camera
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(10, 70)
    j = 0
    while True:
        success, img = cap.read()
        result, object_name = get_objects(img, objects = ['person', 'Animal', 'Small_Animal'])
        # if object identified == person enter loop    
        cv2.imshow('Output', img)
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        elif j == 4:
            break
        if object_name == "[['person']]":
            print("Human identified...")
            cv2.imwrite('detection\\images\\person.png',img)
            cv2.destroyAllWindows() 
            return 1
        elif object_name == "[['Animal']]":
            print("Animal identified...")
            cv2.imwrite('detection\\images\\animal.png',img)
            cv2.destroyAllWindows() 
            return 2 
        elif object_name == "[['Small_Animal']]":
            print("Small Animal identified...")
            cv2.imwrite('detection\\images\\small_animal.png',img)
            cv2.destroyAllWindows() 
            return 3



# https://www.youtube.com/watch?v=hvy9UzMTXpg&t=60s uses pygetwindow check to see if works on rasp pi 
