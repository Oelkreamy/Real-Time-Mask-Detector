"""
FastAPI Mask Detection API - Demo Version
This version can run without model weights for testing deployment
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import json
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🎯 Mask Detection API - Demo",
    description="AI-powered mask detection service (Demo Mode)",
    version="1.0.0",
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

# Class names
class_names = ["incorrect_mask", "with_mask", "without_mask"]

def generate_demo_detections(image_size):
    """Generate fake detections for demo purposes"""
    w, h = image_size
    
    # Generate 1-3 random detections
    num_detections = np.random.randint(1, 4)
    detections = []
    
    for _ in range(num_detections):
        # Random bounding box
        x1 = np.random.randint(0, w // 2)
        y1 = np.random.randint(0, h // 2)
        x2 = x1 + np.random.randint(w // 8, w // 4)
        y2 = y1 + np.random.randint(h // 8, h // 4)
        
        # Random class and confidence
        class_id = np.random.randint(0, 3)
        confidence = np.random.uniform(0.7, 0.95)
        
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(confidence),
            "class_id": int(class_id),
            "class_name": class_names[class_id]
        })
    
    return detections

def draw_demo_detections(image: Image.Image, detections: List[Dict]) -> np.ndarray:
    """Draw detection boxes on image"""
    draw = ImageDraw.Draw(image)
    
    colors = {
        0: "red",      # incorrect_mask
        1: "green",    # with_mask  
        2: "yellow"    # without_mask
    }
    
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        class_id = detection["class_id"]
        confidence = detection["confidence"]
        class_name = detection["class_name"]
        
        # Draw bounding box
        color = colors[class_id]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        draw.text((x1, y1 - 20), label, fill=color)
    
    return np.array(image)

# Health check endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "🎯 Mask Detection API (Demo Mode) is running!",
        "mode": "demo",
        "note": "Using simulated detections for testing"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "mode": "demo",
        "model_loaded": False,
        "demo_mode": True,
        "class_names": class_names,
        "note": "This is a demo version with simulated detections"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and get mask detection predictions (Demo Mode)
    Returns JSON with simulated detection results
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Generate demo detections
        detections = generate_demo_detections(image.size)
        
        # Calculate statistics
        total_detections = len(detections)
        class_counts = {name: 0 for name in class_names}
        
        for detection in detections:
            class_counts[detection['class_name']] += 1
        
        # Calculate compliance rate
        compliance_rate = (class_counts['with_mask'] / total_detections * 100) if total_detections > 0 else 100
        
        return {
            "success": True,
            "mode": "demo",
            "detections": detections,
            "summary": {
                "total_detections": total_detections,
                "class_counts": class_counts,
                "compliance_rate": compliance_rate,
                "image_size": list(image.size)
            },
            "note": "These are simulated detections for demo purposes"
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_image")
async def predict_with_image(file: UploadFile = File(...)):
    """
    Upload an image and get both detection results AND annotated image (Demo Mode)
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Generate demo detections
        detections = generate_demo_detections(image.size)
        
        # Draw detections on image
        annotated_array = draw_demo_detections(image.copy(), detections)
        
        # Convert annotated image to base64
        annotated_pil = Image.fromarray(annotated_array.astype(np.uint8))
        img_buffer = io.BytesIO()
        annotated_pil.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Calculate statistics
        total_detections = len(detections)
        class_counts = {name: 0 for name in class_names}
        
        for detection in detections:
            class_counts[detection['class_name']] += 1
        
        compliance_rate = (class_counts['with_mask'] / total_detections * 100) if total_detections > 0 else 100
        
        return {
            "success": True,
            "mode": "demo",
            "detections": detections,
            "annotated_image": img_base64,
            "summary": {
                "total_detections": total_detections,
                "class_counts": class_counts,
                "compliance_rate": compliance_rate,
                "image_size": list(image.size)
            },
            "note": "These are simulated detections for demo purposes"
        }
        
    except Exception as e:
        logger.error(f"Error during prediction with image: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_stream")
async def predict_stream_image(file: UploadFile = File(...)):
    """
    Upload an image and get annotated image as direct image stream (Demo Mode)
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Generate demo detections
        detections = generate_demo_detections(image.size)
        
        # Draw detections on image
        annotated_array = draw_demo_detections(image, detections)
        
        # Convert to bytes
        annotated_pil = Image.fromarray(annotated_array.astype(np.uint8))
        img_buffer = io.BytesIO()
        annotated_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return StreamingResponse(img_buffer, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error during streaming prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
