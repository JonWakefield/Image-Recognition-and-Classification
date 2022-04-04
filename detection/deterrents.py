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


def siren( x ):
    
    #GPIO setup
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    #GPIO 
    PIR_POWER = 17
    GPIO.setup(PIR_POWER, GPIO.OUT)
    
    if x == 1:
    
        i = 0
        #Pulse alarm
        print("ALARM SOUNDED! PULSING NOW!")
        for i in range(5):
            print('pulse', end = " ")
            GPIO.output(PIR_POWER, HIGH)
            time.sleep(1)
            GPIO.output(PIR_POWER, LOW)
            time.sleep(1)
        
        print(' ')
        
        return 1
    
    return 0