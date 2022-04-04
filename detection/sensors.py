import RPi.GPIO as GPIO
import numpy as np
import argparse
import time
import cv2

HIGH = 1
LOW = 0

#TUNERS
DELAY = 1
TRIGGER = 4

def mic( x ):
    
    #GPIO setup
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    #GPIO
    MIC_DET = 26
    MIC_POWER = 16
    GPIO.setup(MIC_DET, GPIO.IN)
    GPIO.setup(MIC_POWER, GPIO.OUT)
    GPIO.output(MIC_POWER, HIGH)
    
    if x == 1:
        
        MIC_COUNT = 0
        
        print("Starting sound sensing... ")
        time.sleep(DELAY)
        
        while True:
                
            if GPIO.input(MIC_DET):
                print("Sound Detected! COUNT: ", MIC_COUNT)
                MIC_COUNT += 1
                time.sleep(DELAY)

            if MIC_COUNT == TRIGGER:
                GPIO.output(MIC_POWER, LOW)
                MIC_COUNT = 0
                print("Ceasing sound sensing... ")
                time.sleep(DELAY)
                break
            
            #SKIP = input()
            #if SKIP == 'p':
                #return 1
    
        return 1
    
    return 0

def pir( x ):

    #GPIO setup
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    #GPIO 
    PIR_POWER = 4
    PIR_COUNT = [0, 0, 0, 0]   #Y, W, G, P
    PIR_DET = [17, 27, 22, 23] #Y, W, G, P

    GPIO.setup(PIR_POWER, GPIO.OUT)

    for i in range(4):
        GPIO.setup(PIR_DET[i], GPIO.IN)
    
    if x == 1:
            
        s = 1
        GPIO.output(PIR_POWER, HIGH)
        print("PIR sensors booting... will take", s, "seconds to settle")
                
        for d in range(s):
            time.sleep(DELAY)
            print(d, end=" ")
                
        print("Starting motion sensing... ")
        time.sleep(DELAY)

        while True:
                
            if i >= 4: i = 0
                
            if GPIO.input(PIR_DET[i]):
                print("Motion Detected!", " PIR: ", i, " COUNT: ", PIR_COUNT[i])
                PIR_COUNT[i] += 1
                time.sleep(DELAY)

            if PIR_COUNT[i] == TRIGGER:
                GPIO.output(PIR_POWER, LOW)
                PIR_COUNT = [0, 0, 0, 0]
                print("Ceasing motion sensing... ")
                time.sleep(DELAY)
                break
                
            i += 1
            
            #SKIP = input()
            #if SKIP == 'p':
                #return 1
    
        return 1
    
    return 0