import json
import cv2
import numpy as np

with open('test.json','r') as f:
    dict=json.load(f)['result']

img=cv2.imread('test.jpg',0)
for box in dict:
    coordinates=[[int(box['xmin']),int(box['ymin'])],[int(box['xmax']),int(box['ymax'])]]
    for i in range(2):
        for j in range(2):
            coordinates[i][j]=min(coordinates[i][j],320)
    cv2.rectangle(img,*coordinates, 255, -1)
cv2.imwrite('no_components.jpg',img)
