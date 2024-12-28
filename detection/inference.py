import os
import cv2
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device

def load_model(weights='best.pt', device='cuda'):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    return model

def run_inference(model, image_path, conf_thres=0.25, iou_thres=0.45, max_det=1000):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Preprocess
    img_size = model.stride  # or define your own
    img0 = img.copy()
    img = letterbox(img0, auto=True, stride=model.stride, new_shape=model.img_size)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR -> RGB
    img = torch.from_numpy(img).float() / 255.0
    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

    # Scale boxes back
    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(conf),
                    "class": int(cls)
                })

    return detections

if __name__ == '__main__':
    model = load_model(weights='./yolov5/runs/train/coral_model/weights/best.pt', device='cuda')
    test_image = './test_image.jpg'
    results = run_inference(model, test_image)
    print(results)
