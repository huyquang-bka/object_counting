# import cv2
#
# path = r"C:\Users\Admin\Downloads\data ngày 27-11-2021\fullbaidoxe.mp4"
# cap = cv2.VideoCapture(path)
#
# img = cap.read()[1]
# img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
# img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
#
# roi = cv2.selectROIs("image", img)
# print(roi)
# cv2.imshow("image", img)
from fractions import Fraction
print(Fraction("1/12"))
