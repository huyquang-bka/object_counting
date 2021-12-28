rois = []
with open("spot_file/nga_tu_quang_trung.txt", "r") as f:
    lines = f.read().split("\n")
    for line in lines:
        if not line.strip():
            continue
        x, y, w, h = map(int, line.split(","))
        rois.append([x, y, w, h])


def get_line(x, y, w, h):
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    raito = x2 / x1
    y_change = y1 * raito - y2
    b = y_change / (raito - 1)
    a = (y1 - b) / x1
    return a, b


class ObjectCounting():
    def __init__(self, new_dict, old_dict, ROIs=rois):
        self.new_dict = new_dict
        self.old_dict = old_dict
        self.rois = ROIs

    def count(self):
        count = 0
        x, y, w, h = self.rois[0]
        a, b = get_line(x, y, w, h)
        for key, value in self.old_dict:
            x_old, y_old = value
            x_new, y_new = self.new_dict[key]
            if (y_old > a * x_old + b) and (y_new <= a * x_new + b):
                count += 1
        return count

    def count_right(self):
        count = 0
        x, y, w, h = self.rois[0]
        a, b = get_line(x, y, w, h)
        for key, value in self.old_dict:
            try:
                x_old, y_old = value
                x_new, y_new = self.new_dict[key]
                if (x_old < (y_old - b) / a) and (x_new >= (y_new - b) / a):
                    count += 1
            except:
                pass
        return count

    def count_down(self):
        count = 0
        x, y, w, h = self.rois[0]
        a, b = get_line(x, y, w, h)
        for key, value in self.old_dict:
            try:
                x_old, y_old = value
                x_new, y_new = self.new_dict[key]
                if (y_old < a * x_old + b) and (y_new >= a * x_new + b):
                    count += 1
            except:
                pass
        return count

    def count_left(self):
        count = 0
        x, y, w, h = self.rois[0]
        a, b = get_line(x, y, w, h)
        for key, value in self.old_dict:
            try:
                x_old, y_old = value
                x_new, y_new = self.new_dict[key]
                if (x_old > (y_old - b) / a) and (x_new <= (y_new - b) / a):
                    count += 1
            except:
                pass
        return count


def object_counting(new_dict, old_dict, rois=rois):
    count_up = 0
    count_right = 0
    count_down = 0
    count_left = 0

    for index, roi in enumerate(rois):
        x, y, w, h = roi
        a, b = get_line(x, y, w, h)
        for key, value in old_dict.items():
            try:
                x_old, y_old = value
                x_new, y_new = new_dict[key]
                if (y_old > a * x_old + b) and (y_new <= a * x_new + b) and index == 0:
                    count_up += 1
                elif (x_old < (y_old - b) / a) and (x_new >= (y_new - b) / a) and index == 1:
                    count_right += 1
                elif (y_old < a * x_old + b) and (y_new >= a * x_new + b) and index == 2:
                    count_down += 1
                elif (x_old > (y_old - b) / a) and (x_new <= (y_new - b) / a) and index == 3:
                    count_left += 1
            except:
                pass
    return count_up, count_right, count_down, count_left
