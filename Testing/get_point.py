import cv2
from wang_pakage.process_map import is_in_parking_line
from z3 import Real, solve


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        print(x * 2, y * 2)
        print(is_in_parking_line(x * 2, y * 2))
        # with open("../spot_file/parking_spot.csv", "a+") as f:
        #     f.write(f"{int(x * 2)}, {int(y * 2)}\n")
        mouseX, mouseY = x * 2, y * 2


path = r"C:\Users\Admin\Downloads\data ngày 27-11-2021\fullbaidoxe.mp4"
cap = cv2.VideoCapture(path)

img = cap.read()[1]
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
while (1):
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        # spot_dict = {}
        # with open(r"D:\self_project\yolor\spot_file\parking_spot.csv", "r") as f:
        #     for i, line in enumerate(f.readlines()):
        #         if line.strip():
        #             x, y = map(int, line.strip().split(","))
        #             spot_dict[i] = [x, y]
        # for i in range(-1, 3):
        #     x1, y1 = list(spot_dict.values())[i]
        #     x2, y2 = list(spot_dict.values())[i + 1]
        #     a = Real("a")
        #     b = Real("b")
        #     print(f"Case #{i + 2}")
        #     solve(a * x1 + b == y1, a * x2 + b == y2)
        break
