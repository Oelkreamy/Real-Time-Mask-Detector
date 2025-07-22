"""
FastAPI Mask Detection API - Production Version with Cloud Model Loading
Downloads model from cloud storage on startup
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
import requests
import time
from urllib.parse import urlparse

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
    title="🎯 Mask Detection API - Production",
    description="AI-powered mask detection service using YOLOv8",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
device = None
class_names = ["incorrect_mask", "with_mask", "without_mask"]

# Model configuration with cloud storage URLs
MODEL_CONFIG = {
    'CONFIG_PATH': 'model/config/models/yolov8n.yaml',
    'WEIGHTS_URL': os.getenv('MODEL_WEIGHTS_URL', ''),
    'LOCAL_WEIGHTS_PATH': '/tmp/model_weights.pt',
    'IMG_SIZE': (640, 640),
    'CONFIDENCE_THRESHOLD': 0.5
}

def download_model_weights(url: str, local_path: str) -> bool:
    """Download model weights from cloud storage with support for Google Drive, Dropbox, and direct URLs"""
    try:
        logger.info(f"Downloading model weights from {url}")
        
        # Handle different cloud storage services
        if 'drive.google.com' in url:
            logger.info("Detected Google Drive URL")
            if '/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                logger.info(f"Using Google Drive bypass URL for file ID: {file_id}")
            else:
                download_url = url
        elif 'dropbox.com' in url:
            logger.info("Detected Dropbox URL")
            # Convert Dropbox share link to direct download
            if '?dl=0' in url:
                download_url = url.replace('?dl=0', '?dl=1')
                logger.info("Converted Dropbox share link to direct download")
            elif '?dl=1' not in url:
                # Add direct download parameter
                download_url = url + ('?dl=1' if '?' not in url else '&dl=1')
                logger.info("Added direct download parameter to Dropbox URL")
            else:
                download_url = url
        else:
            logger.info("Using direct URL")
            download_url = url
        
        # First request to get the file with retry logic
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Download attempt {attempt + 1}/{max_retries}")
                response = session.get(download_url, headers=headers, stream=True, timeout=300)
                
                # Handle Google Drive virus warning for large files
                if 'drive.google.com' in url and response.status_code == 200 and 'text/html' in response.headers.get('content-type', ''):
                    logger.info("Detected Google Drive virus warning page")
                    content = response.text
                    if 'download_warning' in content or 'virus-scan-warning' in content:
                        import re
                        confirm_match = re.search(r'name="confirm" value="([^"]+)"', content)
                        if confirm_match:
                            confirm_token = confirm_match.group(1)
                            file_id = download_url.split('id=')[1].split('&')[0]
                            bypass_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                            logger.info(f"Retrying with confirm token: {confirm_token}")
                            response = session.get(bypass_url, headers=headers, stream=True, timeout=300)
                
                response.raise_for_status()
                
                # Verify content type
                content_type = response.headers.get('content-type', '').lower()
                logger.info(f"Response Content-Type: {content_type}")
                
                if 'text/html' in content_type:
                    logger.error(f"Still receiving HTML content on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        logger.info("Retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    else:
                        logger.error("All attempts failed - received HTML instead of binary file")
                        return False
                
                # Successful response, break retry loop
                break
                
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    raise
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024 * 5) == 0:  # Log every 5MB
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")
        
        # Verify the downloaded file is a valid PyTorch file
        try:
            with open(local_path, 'rb') as f:
                # Check if it starts with PyTorch magic bytes (PK for zip-like structure)
                magic = f.read(4)
                if not magic.startswith(b'PK'):
                    logger.error("Downloaded file doesn't appear to be a valid PyTorch model")
                    return False
        except Exception as e:
            logger.error(f"Error verifying downloaded file: {e}")
            return False
        
        logger.info(f"✅ Model weights downloaded successfully: {local_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model weights: {e}")
        return False

def load_model():
    """Load the mask detection model"""
    global model, device
    
    try:
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Download model weights if URL is provided
        weights_url = MODEL_CONFIG['WEIGHTS_URL']
        local_weights_path = MODEL_CONFIG['LOCAL_WEIGHTS_PATH']
        
        if weights_url and weights_url.startswith('http'):
            logger.info(f"Attempting to download model from: {weights_url}")
            if not download_model_weights(weights_url, local_weights_path):
                logger.warning("Failed to download model weights, running in fallback mode")
                logger.info("API will provide simulated detections until model is available")
                return False
            weights_path = local_weights_path
        else:
            # Fallback to local path (for development) 
            fallback_paths = [
                'model/weights/yolov8n/best_0.pt',
                'model/weights/converted_yolov8n.pt',
                'yolov8n.pt'
            ]
            weights_path = None
            for path in fallback_paths:
                if os.path.exists(path):
                    weights_path = path
                    logger.info(f"Using local model: {path}")
                    break
            
            if weights_path is None:
                logger.warning("No model weights found locally, running in fallback mode")
                logger.info("API will provide simulated detections until model is available")
                return False
        
        # Load model
        model = DetectionModel(MODEL_CONFIG['CONFIG_PATH'], device=device)
        
        # Load weights
        if os.path.exists(weights_path):
            logger.info(f"Loading weights from {weights_path}")
            state = torch.load(weights_path, map_location=device, weights_only=False)
            
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
            logger.error(f"Weights file not found: {weights_path}")
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
    logger.info("🚀 Starting Production Mask Detection API...")
    success = load_model()
    if not success:
        logger.warning("⚠️ Model loading failed - API running in fallback mode")
    else:
        logger.info("✅ Production API ready to serve requests!")

# Health check endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "🎯 Mask Detection API (Production) is running!",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "mode": "production"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "class_names": class_names,
        "weights_url": MODEL_CONFIG['WEIGHTS_URL'] if MODEL_CONFIG['WEIGHTS_URL'] else "not_configured",
        "mode": "production"
    }

def generate_fallback_detections(image_size):
    """Generate realistic simulated detections when no model is loaded"""
    import random
    
    width, height = image_size
    detections = []
    
    # Generate 1-3 random detections
    num_detections = random.randint(1, 3)
    
    for _ in range(num_detections):
        # Random bounding box
        x1 = random.randint(0, int(width * 0.3))
        y1 = random.randint(0, int(height * 0.3))
        x2 = random.randint(int(width * 0.7), width)
        y2 = random.randint(int(height * 0.7), height)
        
        # Random class and confidence
        class_id = random.randint(0, 2)
        confidence = random.uniform(0.6, 0.9)
        
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": confidence,
            "class_id": class_id,
            "class_name": class_names[class_id]
        })
    
    return detections

def draw_fallback_detections(image: Image.Image, detections) -> np.ndarray:
    """Draw simulated detections on image"""
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Color mapping for different classes
    colors = {
        0: (0, 0, 255),    # incorrect_mask - Red
        1: (0, 255, 0),    # with_mask - Green  
        2: (255, 255, 0)   # without_mask - Yellow
    }
    
    for detection in detections:
        x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
        confidence = detection['confidence']
        class_id = detection['class_id']
        class_name = detection['class_name']
        
        # Get color for this class
        color = colors.get(class_id, (0, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{class_name}: {confidence:.2f} (DEMO)"
        
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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload an image and get mask detection predictions"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Check if model is loaded
        if model is None:
            logger.warning("No model loaded, using fallback mode with simulated detections")
            # Generate simulated detections
            detection_results = generate_fallback_detections(image.size)
            
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
                    "image_size": list(image.size),
                    "mode": "fallback_demo"
                }
            }
        
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
    """Upload an image and get both detection results AND annotated image"""
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
    """Upload an image and get annotated image as direct image stream"""
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
    # Use Render's PORT environment variable (defaults to 8000 for local dev)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
