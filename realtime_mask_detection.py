
import cv2
import torch
import numpy as np
from math import ceil
from matplotlib import colormaps as cm
from model.models.detection_model import DetectionModel
from model.data.detections import Detections
from model.data.utils import pad_to, unpad

# --- CONFIG ---
MODEL_CONFIG = "model/config/models/yolov8n.yaml"
WEIGHTS = "model/weights/yolov8n/best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (640, 640)
CLASS_NAMES = ['with_mask', 'without_mask', 'mask_worn_incorrectly']  # Adjust to your dataset

# --- Load Model ---
model = DetectionModel(MODEL_CONFIG, device=DEVICE)
model.load(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()
model.mode = 'eval'

# --- Visualization setup ---
cmap = cm['jet']
num_classes = len(CLASS_NAMES)

# --- Webcam Stream ---
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Could not open webcam."

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h0, w0 = frame_rgb.shape[:2]

    # Resize to preserve aspect ratio
    ratio = min(IMG_SIZE[0] / h0, IMG_SIZE[1] / w0)
    h, w = min(ceil(h0 * ratio), IMG_SIZE[0]), min(ceil(w0 * ratio), IMG_SIZE[1])
    image_resized = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert and pad
    image_tensor = torch.from_numpy(image_resized.transpose((2, 0, 1))).float() / 255.0
    image_tensor, pads = pad_to(image_tensor, shape=IMG_SIZE)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        preds = model(image_tensor)[0]

    detections = Detections.from_yolo(preds)
    detections.unpad_xyxy(pads)
    detections.view(frame, classes_dict={i: c for i, c in enumerate(CLASS_NAMES)}, cmap=cmap, num_classes=num_classes)

    cv2.imshow("YOLOv8 Real-Time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
