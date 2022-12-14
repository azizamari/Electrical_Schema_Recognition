import json
import cv2
import numpy as np

with open('test.json','r') as f:
    dict=json.load(f)['result']

blank_img=np.zeros(shape=(320,320))
for box in dict:
    coordinates=[[int(box['xmin'])-5,int(box['ymin'])-5],[int(box['xmax'])+5,int(box['ymax'])+5]]
    for i in range(2):
        for j in range(2):
            coordinates[i][j]=min(coordinates[i][j],320)
    cv2.rectangle(blank_img,*coordinates, 255, 4)
cv2.imwrite('boxes.jpg',blank_img)
