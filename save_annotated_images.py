import os
import torch
import cv2
from matplotlib import colormaps as cm
from model.models.detection_model import DetectionModel
from model.data.dataset import Dataset
from model.data.detections import Detections
from torch.utils.data import DataLoader
from model.data.utils import unpad

# --- CONFIG ---
MODEL_CONFIG = "model/config/models/yolov8n.yaml"
WEIGHTS = "model/weights/yolov8n/best.pt"  # Change to your checkpoint
DATASET_CONFIG = "model/config/datasets/mask.yaml"
DATASET_MODE = "train"  # or "train" or "test"
DEVICE = "cuda"  # or "cpu"

# Load model
model = DetectionModel(MODEL_CONFIG, device=DEVICE)
model.load(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()
model.mode = 'eval'

# Load dataset
dataset = Dataset(DATASET_CONFIG, mode=DATASET_MODE)
dataloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=True, collate_fn=Dataset.collate_fn)

cmap = cm['jet']

# Prepare output directory
save_path = os.path.join(os.path.dirname(DATASET_CONFIG), dataset.config['path'], 'results', DATASET_MODE, 'images')
os.makedirs(save_path, exist_ok=True)

for batch in dataloader:
    with torch.no_grad():
        preds = model(batch['images'].to(DEVICE))

    for i in range(len(preds)):
        detections = Detections.from_yolo(preds[i])
        # Prepare image for drawing (denormalize to 0-255 and convert to uint8)
        image = batch['images'][i].detach().cpu().numpy().transpose((1, 2, 0))
        image = (image * 255.0).clip(0, 255).astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        detections.view(image, classes_dict=dataset.config['names'], cmap=cmap)

        # Remove letterbox padding if present
        if 'orig_shapes' in batch and 'padding' in batch:
            orig_h, orig_w = batch['orig_shapes'][i]
            pad = batch['padding'][i]  # (pad_left, pad_top, pad_right, pad_bottom)
            if any(pad):
                image_t = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
                image_t = unpad(image_t, pad)
                image = image_t.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
                if (image.shape[0], image.shape[1]) != (orig_h, orig_w):
                    image = cv2.resize(image, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Save annotated image
        out_img_path = os.path.join(save_path, batch['ids'][i] + ".jpg")
        cv2.imwrite(out_img_path, image)
        print(f"Saved: {out_img_path}")

print(f"All annotated images saved to: {save_path}")
