import cv2
import numpy as np


path = r"C:\Users\Admin\Downloads\data ngày 27-11-2021\fullbaidoxe.mp4"
cap = cv2.VideoCapture(path)

img = cap.read()[1]
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

img_copy = np.copy(img)

cv2.imshow("img_copy", img_copy)

input_pts = np.float32([[404, 176], [634, 232], [774, 1560], [208, 1564]])
output_pts = np.float32([[0, 0], [1080, 0], [1080, 1920], [0, 1920]])

M = cv2.getPerspectiveTransform(input_pts, output_pts)
out = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

cv2.imshow("out", cv2.resize(out, dsize=None, fx=0.5, fy=0.5))
cv2.waitKey()
