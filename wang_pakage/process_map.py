# from z3 import solve, Real
from fractions import Fraction

line_dict = {}

with open("D:\self_project\yolor\spot_file\parking_line.csv", "r") as f:
    for i, line in enumerate(f.readlines()):
        if line.strip():
            a, b = map(Fraction, line.strip().split(","))
            line_dict[i] = [a, b]


def is_in_parking_line(x, y):
    for key, value in line_dict.items():
        b, a = value
        if key in [0, 1, 3]:
            if y <= a * x + b:
                return False
        else:
            if y >= a * x + b:
                return False
    return True
