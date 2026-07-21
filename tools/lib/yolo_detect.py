import numpy as np
from ultralytics import YOLO


def detect_objects_in_frame(model, frame, conf_thres=0.8, iou_thres=0.45):
    results = model(frame, conf=conf_thres, iou=iou_thres)[0]
    detections = results.obb.xywhr.cpu().numpy()
    scores = results.obb.conf.cpu().numpy()
    class_ids = results.obb.cls.cpu().numpy().astype(int)
    class_names = [model.names[i] for i in class_ids]
    return [
        ((u, v, w, h, r), score, class_id, class_name)
        for (u, v, w, h, r), score, class_id, class_name in zip(
            detections, scores, class_ids, class_names, strict=True
        )
    ]


def load_model(model_path, device=""):
    model = YOLO(model_path)
    if device:
        model.to(device)
    return model
