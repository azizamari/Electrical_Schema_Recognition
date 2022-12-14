import cv2

boxes_img=cv2.imread('boxes.jpg',0)
circuit_img=cv2.imread('test.jpg',0)

circuit_img=cv2.bitwise_not(circuit_img)
result=cv2.bitwise_and(circuit_img,boxes_img)
# result=cv2.bitwise_not(result)
cv2.imwrite('result.jpg',result)
