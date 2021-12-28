# import cv2
# import time
#
# cap = cv2.VideoCapture(r"C:\Users\Admin\Downloads\data ngày 27-11-2021\nhandienbiensoxe.mp4")
#
# cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
# count = 0
# while True:
#     count += 1
#     ret, frame = cap.read()
#
#     cv2.imshow("image", frame)
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break
#     if count == 25:
#         time.sleep(4)
#         count += 4 * 25
#         cap.set(cv2.CAP_PROP_POS_FRAMES, count)

import tensorflow as tf

print(tf.__version__)

