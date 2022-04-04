import numpy as np
from PIL import Image
import cv2
import os.path

def image_man( state ):

    print("Entering image cropping...")

    # down sample image to (32x32)
    def downsample(image):
        img_size = 32
        image_ds = cv2.imread(image)
        image_ds = cv2.resize(image_ds , ( img_size, img_size))
        return image_ds



    # person
    if state == 1:
        try:
            image = Image.open("detection\\images\\person.png").convert("RGB")
        except:
            print("Could not find file...")
            return 0
        img_array = np.array(image)
        original = img_array.copy() # save original (could remove)

        # # Find X,Y coordinates of all green pixels
        Y_green, X_green = np.where(np.all(img_array==[0 , 255 , 0 ],axis=2))

        yMin, yMax = Y_green[0] , Y_green[-1]
        xMin, xMax = X_green[0] , X_green[-1]

        ROI = original[yMin: yMax, xMin:xMax]    

        Image.fromarray(ROI).save('detection\\images\\cropped_person.png')

        print("Image successfully cropped...")

        # path to cropped image
        image = 'detection\\images\\cropped_person.png'
        image = downsample(image)
        Image.fromarray(image).save('detection\\images\\downsample_person.png')
        print("Image successfully down sampled ... ")
        return 1



    # animal
    if state == 2:
        try:
            image = Image.open("detection\\images\\animal.png").convert("RGB")
        except:
            print("Could not find file...")
            return 0

        img_array = np.array(image)

        # # Find X,Y coordinates of all green pixels
        Y_green, X_green = np.where(np.all(img_array==[0 , 255 , 0 ],axis=2))

        yMin, yMax = Y_green[0] , Y_green[-1]
        xMin, xMax = X_green[0] , X_green[-1]

        ROI = img_array[yMin: yMax, xMin:xMax]    

        Image.fromarray(ROI).save('detection\\images\\cropped_animal.png')

        print("Image successfully cropped...")

        # path to cropped image
        image = 'detection\\images\\cropped_animal.png'
        image = downsample(image)
        Image.fromarray(image).save('detection\\images\\downsample_animal.png')
        print("Image successfully down sampled ... ")
        return 2


    # small animal
    if state == 3:
        try:
            image = Image.open("detection\\images\\small_animal.png").convert("RGB")
        except:
            print("Could not find file...")
            return 0
        img_array = np.array(image)

        # # Find X,Y coordinates of all green pixels
        Y_green, X_green = np.where(np.all(img_array==[0 , 255 , 0 ],axis=2))

        yMin, yMax = Y_green[0] , Y_green[-1]
        xMin, xMax = X_green[0] , X_green[-1]

        ROI = img_array[yMin: yMax, xMin:xMax]    

        Image.fromarray(ROI).save('detection\\images\\cropped_small_animal.png')

        print("Image successfully cropped...")

        # path to cropped image
        image = 'detection\\images\\cropped_small_animal.png'
        image = downsample(image)
        Image.fromarray(image).save('detection\\images\\downsample_small_animal.png')
        print("Image successfully down sampled ... ")
        return 3
