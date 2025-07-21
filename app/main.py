"""
FastAPI Mask Detection API
Provides endpoints for image upload and mask detection inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
from pathlib import Path
import sys
import os
from typing import List, Dict, Any
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import your model components
from model.models.detection_model import DetectionModel
from model.data.detections import Detections
from model.data.utils import pad_to

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🎯 Mask Detection API",
    description="AI-powered mask detection service using YOLOv8",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
device = None
class_names = ["incorrect_mask", "with_mask", "without_mask"]

# Model configuration
MODEL_CONFIG = {
    'CONFIG_PATH': 'model/config/models/yolov8n.yaml',
    'WEIGHTS_PATH': 'model/weights/yolov8n/best_0.pt',
    'IMG_SIZE': (640, 640),
    'CONFIDENCE_THRESHOLD': 0.5
}

def load_model():
    """Load the mask detection model"""
    global model, device
    
    try:
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model
        model = DetectionModel(MODEL_CONFIG['CONFIG_PATH'], device=device)
        
        # Load weights
        if os.path.exists(MODEL_CONFIG['WEIGHTS_PATH']):
            logger.info(f"Loading weights from {MODEL_CONFIG['WEIGHTS_PATH']}")
            state = torch.load(MODEL_CONFIG['WEIGHTS_PATH'], map_location=device)
            
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
                logger.info(f"Loaded checkpoint from epoch {state.get('epoch', 'unknown')}")
            else:
                model.load(state)
                logger.info("Loaded model state dict")
            
            model.eval()
            model.mode = 'eval'
            
            logger.info("✅ Model loaded successfully")
            return True
        else:
            logger.error(f"Weights file not found: {MODEL_CONFIG['WEIGHTS_PATH']}")
            return False
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def preprocess_image(image: Image.Image, img_size=(640, 640)):
    """Preprocess image for model inference"""
    # Convert PIL to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert BGR to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h0, w0 = image_rgb.shape[:2]
    
    # Resize maintaining aspect ratio
    ratio = min(img_size[0] / h0, img_size[1] / w0)
    h, w = int(h0 * ratio), int(w0 * ratio)
    image_resized = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor and pad
    image_tensor = torch.from_numpy(image_resized.transpose((2, 0, 1))).float() / 255.0
    image_tensor, pads = pad_to(image_tensor, shape=img_size)
    
    return image_tensor.unsqueeze(0), pads, (w0, h0), (w, h)

def postprocess_detections(detections, pads, original_size, resized_size, confidence_threshold=0.5):
    """Postprocess model detections"""
    # Unpad detections
    detections.unpad_xyxy(pads)
    
    # Scale back to original image size
    w0, h0 = original_size
    w, h = resized_size
    scale_x = w0 / w
    scale_y = h0 / h
    
    if hasattr(detections, 'xyxy') and detections.xyxy is not None and len(detections.xyxy) > 0:
        detections.xyxy[:, [0, 2]] *= scale_x  # x coordinates
        detections.xyxy[:, [1, 3]] *= scale_y  # y coordinates
        
        # Apply confidence filtering
        if hasattr(detections, 'confidence') and detections.confidence is not None:
            mask = detections.confidence >= confidence_threshold
            detections.xyxy = detections.xyxy[mask]
            detections.confidence = detections.confidence[mask]
            if hasattr(detections, 'class_id'):
                detections.class_id = detections.class_id[mask]
    
    return detections

def draw_detections(image: Image.Image, detections, class_names: List[str]) -> np.ndarray:
    """Draw detection boxes and labels on image"""
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    if not hasattr(detections, 'xyxy') or detections.xyxy is None or len(detections.xyxy) == 0:
        return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Color mapping for different classes
    colors = {
        0: (0, 0, 255),    # incorrect_mask - Red
        1: (0, 255, 0),    # with_mask - Green  
        2: (255, 255, 0)   # without_mask - Yellow
    }
    
    for i in range(len(detections.xyxy)):
        # Handle both tensor and numpy array formats
        if hasattr(detections.xyxy[i], 'int'):
            x1, y1, x2, y2 = detections.xyxy[i].int().tolist()
        else:
            x1, y1, x2, y2 = detections.xyxy[i].astype(int).tolist()
        
        # Handle confidence
        if hasattr(detections.confidence[i], 'item'):
            confidence = detections.confidence[i].item()
        else:
            confidence = float(detections.confidence[i])
        
        # Handle class_id
        if hasattr(detections, 'class_id') and detections.class_id is not None:
            if hasattr(detections.class_id[i], 'item'):
                class_id = detections.class_id[i].item()
            else:
                class_id = int(detections.class_id[i])
        else:
            class_id = 0
        
        # Get color for this class
        color = colors.get(class_id, (0, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{class_names[class_id]}: {confidence:.2f}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            image_np, 
            (x1, y1 - text_height - baseline - 5), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            image_np, label, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

def process_detections_to_json(detections) -> List[Dict[str, Any]]:
    """Convert detections to JSON format"""
    results = []
    
    if not hasattr(detections, 'xyxy') or detections.xyxy is None or len(detections.xyxy) == 0:
        return results
    
    for i in range(len(detections.xyxy)):
        # Handle coordinates
        if hasattr(detections.xyxy[i], 'tolist'):
            x1, y1, x2, y2 = detections.xyxy[i].tolist()
        else:
            x1, y1, x2, y2 = detections.xyxy[i].astype(float).tolist()
        
        # Handle confidence
        if hasattr(detections.confidence[i], 'item'):
            confidence = detections.confidence[i].item()
        else:
            confidence = float(detections.confidence[i])
        
        # Handle class_id
        if hasattr(detections, 'class_id') and detections.class_id is not None:
            if hasattr(detections.class_id[i], 'item'):
                class_id = detections.class_id[i].item()
            else:
                class_id = int(detections.class_id[i])
        else:
            class_id = 0
        
        results.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
            "class_id": class_id,
            "class_name": class_names[class_id]
        })
    
    return results

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("🚀 Starting Mask Detection API...")
    success = load_model()
    if not success:
        logger.error("❌ Failed to load model on startup!")
    else:
        logger.info("✅ API ready to serve requests!")

# Health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "🎯 Mask Detection API is running!",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "class_names": class_names,
        "model_config": MODEL_CONFIG
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and get mask detection predictions
    Returns JSON with detection results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess image
        image_tensor, pads, original_size, resized_size = preprocess_image(image)
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            preds = model(image_tensor)[0]
        
        # Process detections
        detections = Detections.from_yolo(preds)
        detections = postprocess_detections(
            detections, pads, original_size, resized_size, 
            MODEL_CONFIG['CONFIDENCE_THRESHOLD']
        )
        
        # Convert to JSON format
        detection_results = process_detections_to_json(detections)
        
        # Calculate statistics
        total_detections = len(detection_results)
        class_counts = {name: 0 for name in class_names}
        
        for detection in detection_results:
            class_counts[detection['class_name']] += 1
        
        # Calculate compliance rate
        compliance_rate = (class_counts['with_mask'] / total_detections * 100) if total_detections > 0 else 100
        
        return {
            "success": True,
            "detections": detection_results,
            "summary": {
                "total_detections": total_detections,
                "class_counts": class_counts,
                "compliance_rate": compliance_rate,
                "image_size": original_size
            }
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_image")
async def predict_with_image(file: UploadFile = File(...)):
    """
    Upload an image and get both detection results AND annotated image
    Returns JSON with detections + base64 encoded annotated image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess image
        image_tensor, pads, original_size, resized_size = preprocess_image(image)
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            preds = model(image_tensor)[0]
        
        # Process detections
        detections = Detections.from_yolo(preds)
        detections = postprocess_detections(
            detections, pads, original_size, resized_size,
            MODEL_CONFIG['CONFIDENCE_THRESHOLD']
        )
        
        # Draw detections on image
        annotated_image = draw_detections(image, detections, class_names)
        
        # Convert annotated image to base64
        annotated_pil = Image.fromarray(annotated_image)
        img_buffer = io.BytesIO()
        annotated_pil.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Convert detections to JSON
        detection_results = process_detections_to_json(detections)
        
        # Calculate statistics
        total_detections = len(detection_results)
        class_counts = {name: 0 for name in class_names}
        
        for detection in detection_results:
            class_counts[detection['class_name']] += 1
        
        compliance_rate = (class_counts['with_mask'] / total_detections * 100) if total_detections > 0 else 100
        
        return {
            "success": True,
            "detections": detection_results,
            "annotated_image": img_base64,
            "summary": {
                "total_detections": total_detections,
                "class_counts": class_counts,
                "compliance_rate": compliance_rate,
                "image_size": original_size
            }
        }
        
    except Exception as e:
        logger.error(f"Error during prediction with image: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_stream")
async def predict_stream_image(file: UploadFile = File(...)):
    """
    Upload an image and get annotated image as direct image stream
    Returns the annotated image directly (for easy display)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess image
        image_tensor, pads, original_size, resized_size = preprocess_image(image)
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            preds = model(image_tensor)[0]
        
        # Process detections
        detections = Detections.from_yolo(preds)
        detections = postprocess_detections(
            detections, pads, original_size, resized_size,
            MODEL_CONFIG['CONFIDENCE_THRESHOLD']
        )
        
        # Draw detections on image
        annotated_image = draw_detections(image, detections, class_names)
        
        # Convert to bytes
        annotated_pil = Image.fromarray(annotated_image)
        img_buffer = io.BytesIO()
        annotated_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return StreamingResponse(img_buffer, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error during streaming prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # For local network access, use your actual IP or 0.0.0.0
    # For localhost only, use "127.0.0.1" 
    uvicorn.run(app, host="0.0.0.0", port=8000)
