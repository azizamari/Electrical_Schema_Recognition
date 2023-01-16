import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import imutils

path='data/collected_image'
output='data/'
for count,img_path in enumerate(os.listdir(path)):

    img=cv2.imread(os.path.join(path,img_path), cv2.IMREAD_GRAYSCALE)

    img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,6)
    # _,img=cv2.threshold(img,160,250,cv2.THRESH_BINARY)

    # # resize without losing aspect ratio
    if img.shape[0]>img.shape[1]:
        img=imutils.resize(img, height=320)
        matrix=np.ones((320,320-img.shape[1]))*255
        img=np.hstack((img,matrix))

    else:
        img=imutils.resize(img, width=320)
        matrix=np.ones((320-img.shape[0],320))*255
        img=np.vstack((img,matrix))


    # img=cv2.resize(img,size,interpolation = cv2.INTER_AREA)

    # # fill missing values by white pixels to make sure image size is 320x320
    # img=img-255
    # img.resize((320,320))
    # img=img+255
    
    
    #save img
    cv2.imwrite(os.path.join(output,img_path),img)