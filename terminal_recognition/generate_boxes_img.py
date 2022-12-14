import json
import cv2
import numpy as np

with open('test.json','r') as f:
    dict=json.load(f)['result']

blank_img=np.zeros(shape=(320,320))
for box in dict:
    coordinates=[(int(box['xmin']),int(box['ymin'])),(int(box['xmax']),int(box['ymax']))]
    cv2.rectangle(blank_img,*coordinates, 255, 1)
cv2.imwrite('result.jpg',blank_img)
