
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
WEIGHTS = "model/weights/yolov8n/best_0.pt"  # Use the properly converted weights!
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (640, 640)
CLASS_NAMES = ["incorrect_mask", "with_mask", "without_mask"]  # Your mask detection classes
CONFIDENCE_THRESHOLD = 0.25  # Lower threshold to catch more detections initially
NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold

# --- Load Model ---
print(f"🔥 Loading model on {DEVICE}...")
model = DetectionModel(MODEL_CONFIG, device=DEVICE)

# Smart checkpoint loading
checkpoint = torch.load(WEIGHTS, map_location=DEVICE)
print(f"📂 Loading weights from: {WEIGHTS}")

# Check if this is a training checkpoint or direct state dict
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    # This is a training checkpoint - extract the model weights
    print("✅ Loading from training checkpoint (with model_state_dict)")
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    
    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"📊 Checkpoint from epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"📊 Validation loss: {checkpoint['val_loss']:.4f}")
    if 'train_loss' in checkpoint:
        print(f"📊 Training loss: {checkpoint['train_loss']:.4f}")
        
elif isinstance(checkpoint, dict) and 'ema_state_dict' in checkpoint:
    # Use EMA weights for better accuracy
    print("✅ Loading from EMA weights (better accuracy)")
    model.load_state_dict(checkpoint['ema_state_dict'], strict=True)
    
else:
    # Direct state dict format
    print("✅ Loading from direct state dict")
    model.load_state_dict(checkpoint, strict=True)
model.eval()
model.mode = 'eval'
print(f"🎯 Model loaded successfully and ready for inference!")

# --- Visualization setup ---
cmap = cm['jet']
num_classes = len(CLASS_NAMES)

# --- Webcam Stream ---
print("🎥 Initializing webcam...")
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "❌ Could not open webcam. Check if camera is connected."

# Keep default resolution to avoid coordinate issues
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FPS, 30)

print("🎯 Real-time mask detection started!")
print("📋 Controls:")
print("   - Press 'q' to quit")
print("   - Press 's' to save current frame")
print("   - Press 'c' to adjust confidence threshold")
print("="*50)

# Detection smoothing
prev_detections = []
detection_history_size = 3
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam")
        break

    frame_count += 1
    
    # Process every frame for smooth detection
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
    
    # Debug: Print detection attributes on first frame
    if frame_count == 1:
        print(f"🔍 Detection attributes: {dir(detections)}")
        print(f"🔍 Has confidence: {hasattr(detections, 'confidence')}")
        if hasattr(detections, 'confidence'):
            print(f"🔍 Confidence type: {type(detections.confidence)}")
    
    # Apply confidence filtering to reduce flickering
    try:
        if hasattr(detections, 'confidence') and detections.confidence is not None and len(detections.confidence) > 0:
            mask = detections.confidence >= CONFIDENCE_THRESHOLD
            # Manually filter detections based on confidence
            if hasattr(detections, 'xyxy') and detections.xyxy is not None:
                detections.xyxy = detections.xyxy[mask]
            if hasattr(detections, 'confidence'):
                detections.confidence = detections.confidence[mask]
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                detections.class_id = detections.class_id[mask]
    except Exception as e:
        # If filtering fails, continue with original detections
        if frame_count <= 5:  # Only print first few errors
            print(f"⚠️ Confidence filtering failed: {e}")
    
    detections.unpad_xyxy(pads)
    
    # Scale detections back to original frame size
    scale_x = w0 / w
    scale_y = h0 / h
    if hasattr(detections, 'xyxy') and detections.xyxy is not None:
        detections.xyxy[:, [0, 2]] *= scale_x  # x coordinates
        detections.xyxy[:, [1, 3]] *= scale_y  # y coordinates
    
    detections.view(frame, classes_dict={i: c for i, c in enumerate(CLASS_NAMES)}, cmap=cmap, num_classes=num_classes)

    # Add frame info and confidence threshold
    cv2.putText(frame, f"Frame: {frame_count} | Device: {DEVICE} | Conf: {CONFIDENCE_THRESHOLD:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Show detection count
    num_detections = len(detections.xyxy) if hasattr(detections, 'xyxy') and detections.xyxy is not None else 0
    cv2.putText(frame, f"Detections: {num_detections}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("🎯 YOLOv8 Mask Detection - Advanced Training", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("🛑 Quitting...")
        break
    elif key == ord('s'):
        # Save current frame
        filename = f"detection_frame_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"💾 Saved frame as {filename}")
    elif key == ord('c'):
        # Adjust confidence threshold
        print(f"\nCurrent confidence threshold: {CONFIDENCE_THRESHOLD}")
        try:
            new_conf = float(input("Enter new confidence threshold (0.1-0.9): "))
            if 0.1 <= new_conf <= 0.9:
                CONFIDENCE_THRESHOLD = new_conf
                print(f"✅ Confidence threshold updated to {CONFIDENCE_THRESHOLD}")
            else:
                print("❌ Invalid range. Keeping current value.")
        except ValueError:
            print("❌ Invalid input. Keeping current value.")

cap.release()
cv2.destroyAllWindows()
print("✅ Real-time detection completed!")
print(f"📊 Total frames processed: {frame_count}")
