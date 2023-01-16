import cv2
import numpy as np

def draw_lines(img, lines, color = [255, 0, 0], thickness = 1):
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

img=cv2.imread('no_components.jpg',0)
img=cv2.bitwise_not(img)


rho = 1
theta = np.pi / 180
threshold = 70
min_line_len = 4
max_line_gap = 5

dst = cv2.Canny(img, 150, 200, None, 3)

lines = cv2.HoughLinesP(
    dst, rho, theta, threshold, minLineLength = min_line_len, maxLineGap = max_line_gap)

# Draw all lines found onto a new image.
hough = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
draw_lines(hough, lines)

print("Found {} lines, including: {}".format(len(lines), lines[0]))
cv2.imwrite('result.jpg', hough)