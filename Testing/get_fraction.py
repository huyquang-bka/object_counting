from z3 import Real, solve

spot_dict = {}
with open("D:\self_project\yolor\spot_file\parking_spot.csv", "r") as f:
    for i, line in enumerate(f.readlines()):
        if line.strip():
            x, y = map(int, line.strip().split(","))
            spot_dict[i] = [x, y]

if __name__ == "__main__":
    for i in range(-1, 3):
        x1, y1 = list(spot_dict.values())[i]
        x2, y2 = list(spot_dict.values())[i + 1]
        a = Real("a")
        b = Real("b")
        print(f"Case #{i + 2}")
        solve(a * x1 + b == y1, a * x2 + b == y2)
