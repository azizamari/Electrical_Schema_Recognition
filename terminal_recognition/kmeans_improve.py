# using kmeans to find centroid for intersection points
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = np.float32(cv2.imread("result.jpg",0))
K = 10
attempts=10
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(img,K,None,criteria ,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
print(label[label!=0])
res = center[label.flatten()]
print(len(np.argwhere(res==1)))


result_image1 = res.reshape((320,320))

cv2.imwrite('kmeans.jpg',result_image1)