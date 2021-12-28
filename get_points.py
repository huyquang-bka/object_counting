import cv2
import numpy as np

from Testing.get_fraction import spot_dict

ls = []


# click event function
def click_event(event, x, y, flags, param):
    # global gl_x, gl_y
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ",", y)
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        # gl_x = x
        # gl_y = y
        ls.append(x, y)
        # cv2.imshow("image", img)


# Here, you need to change the image name and it's path according to your directory
path = r"C:\Users\Admin\Downloads\CAMERA NGÃ TƯ QUANG TRUNG - NGUYỄN THỊ MINH KHAI - YouTube - Cốc Cốc 2021-12-15 15-35-19.mp4"
img = cv2.VideoCapture(path).read()[1]
# calling the mouse click event
# cv2.setMouseCallback("image", click_event)
# cv2.imshow("image", img)
# key = cv2.waitKey(0)
# while True:
#     cv2.imshow("image", img)
#     key = cv2.waitKey(0)
#     if key == ord("q"):
#         break
#     if key == ord("s"):
#         with open("points.txt", "a+") as f:
#             f.write(f"{ls}")
#     if key == ord("c"):
#         ls.remove(ls[-1])
rois = cv2.selectROIs("image", img, showCrosshair=True)
# with open("spot_file/nga_tu", "r") as f:
#     lines = f.read().split("\n")
#     rois = []
#     for line in lines:
#         if not line.strip():
#             continue
#         x, y, w, h = map(int, line.split(","))
#         rois.append([x, y, w, h])
f = open("spot_file/nga_tu_quang_trung.txt", "w+")
for r in rois:
    x, y, w, h = r
    f.write(f"{x},{y},{w},{h}\n")
    cv2.line(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow("image", img)
key = cv2.waitKey(0)
print(rois)
