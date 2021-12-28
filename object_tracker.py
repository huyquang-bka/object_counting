import os

import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from utils.torch_utils import select_device

from models.models import *
from utils.datasets import *
from utils.general import *
import time
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from utils.plots import plot_one_box
from wang_pakage.counting import object_counting, rois

import torch

torch.cuda.set_per_process_memory_fraction(0.5, 0)

import opt


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


# initialize deepsort
cfg = get_config()
cfg.merge_from_file(opt.config_deepsort)
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

out, source, weights, view_img, save_txt, imgsz, cfg, names = \
    opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names

# Initialize
device = select_device(opt.device, batch_size=32)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = Darknet(cfg, imgsz).cuda()
model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
# model = attempt_load(weights, map_location=device)  # load FP32 model
# imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
model.to(device).eval()
if half:
    model.half()  # to FP16

# Get names and colors
names = load_classes(names)
colors = (0, 0, 255)


# Run inference
# img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
# _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once


def detect(img0):
    H, W, _ = img0.shape
    img = letterbox(img0, new_shape=imgsz, auto_size=64)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0 = img0.copy()

        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            id_dict = {}
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            xywhs, confs, clss = [], [], []
            for *xyxy, conf, cls in det:
                label = names[int(cls)]
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                xywhs.append([x1, y1, x2, y2])
                confs.append(conf)
                clss.append(cls)
            xywhs = xyxy2xywh(torch.Tensor(xywhs))
            confs = torch.Tensor(confs)
            clss = torch.tensor(clss)
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)

            # draw boxes for visualization
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]
                    c = int(cls)  # integer class
                    x_center = (bboxes[0] + bboxes[2]) / 2
                    y_center = (bboxes[1] + bboxes[3]) / 2
                    id_dict[id] = [x_center, y_center]
                    label = str(id)
                    color = compute_color_for_id(id)
                    plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)
        # else:
        #     deepsort.increment_ages()
    try:
        return im0, id_dict
    except:
        return img0, None


if __name__ == '__main__':
    path = r"C:\Users\Admin\Downloads\CAMERA NGÃ TƯ QUANG TRUNG - NGUYỄN THỊ MINH KHAI - YouTube - Cốc Cốc 2021-12-15 15-35-19.mp4"
    cap = cv2.VideoCapture(path)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    fourcc = 'mp4v'  # output video codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter("demo_counting.mp4", cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    with torch.no_grad():
        frame = 0
        old_id_dict = {}
        count_obj_down = 0
        count_obj_up = 0
        count_obj_right = 0
        count_obj_left = 0
        while True:
            frame += 1
            t = time.time()
            ret, img0 = cap.read()
            H, W, _ = img0.shape
            H_limit = 3 * H // 5
            img, id_dict = detect(img0)
            if id_dict is not None:
                count_up, count_right, count_down, count_left = object_counting(new_dict=id_dict, old_dict=old_id_dict)
                count_obj_down += count_down
                count_obj_up += count_up
                count_obj_right += count_right
                count_obj_left += count_left
                old_id_dict = id_dict
            cv2.putText(img, f"Motorbike counter down: {count_obj_down}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(img, f"Motorbike counter up: {count_obj_up}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(img, f"Motorbike counter right: {count_obj_right}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(img, f"Motorbike counter left: {count_obj_left}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            for r in rois:
                x, y, w, h = r
                cv2.line(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            vid_writer.write(img)
            # cv2.imshow("Image", img)
            # key = cv2.waitKey(1)
            print(f"Frame: {frame}/{total_frame}")
            # print("FPS: ", 1 // (time.time() - t))
            # if key == ord("q"):
            #     break
