# Mask Detection API

A FastAPI-based REST API for real-time mask detection using YOLOv8.

## 🚀 Quick Start

### Local Development

1. **Install dependencies**:
```bash
pip install -r app/requirements.txt
```

2. **Run the API**:
```bash
cd app
python main.py
```

3. **Test the API**:
```bash
python app/test_api.py
```

### Docker Development

1. **Build and run with Docker Compose**:
```bash
docker-compose up --build
```

2. **Access the API**:
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## 📋 API Endpoints

### Health Check
- **GET** `/` - Basic health check
- **GET** `/health` - Detailed health information

### Prediction Endpoints
- **POST** `/predict` - Upload image, get JSON detection results
- **POST** `/predict_image` - Upload image, get JSON + annotated image (base64)
- **POST** `/predict_stream` - Upload image, get annotated image directly

## 📁 API Response Format

### `/predict` Response:
```json
{
  "success": true,
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "class_id": 1,
      "class_name": "with_mask"
    }
  ],
  "summary": {
    "total_detections": 3,
    "class_counts": {
      "with_mask": 2,
      "without_mask": 1,
      "incorrect_mask": 0
    },
    "compliance_rate": 66.7,
    "image_size": [1920, 1080]
  }
}
```

## 🐳 Docker Deployment

### Build Docker Image:
```bash
docker build -t mask-detection-api .
```

### Run Docker Container:
```bash
docker run -p 8000:8000 mask-detection-api
```

## ☁️ Cloud Deployment Options

### 1. Railway (Recommended)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### 2. Render
1. Connect your GitHub repo to Render
2. Create a new Web Service
3. Use these settings:
   - Build Command: `pip install -r app/requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### 3. DigitalOcean App Platform
1. Connect your GitHub repo
2. Create a new App
3. Configure as Python app with start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### 4. AWS/GCP/Azure
Use their container services with the Docker image.

## 🧪 Testing

### Test with cURL:
```bash
# Health check
curl http://localhost:8000/health

# Upload image for prediction
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

### Test with Python:
```python
import requests

# Test prediction
with open('test_image.jpg', 'rb') as f:
    files = {'file': ('image.jpg', f, 'image/jpeg')}
    response = requests.post('http://localhost:8000/predict', files=files)
    print(response.json())
```

## 🔧 Configuration

Edit `app/main.py` to modify:
- Model paths
- Confidence threshold
- Image size
- CORS settings

## 📝 Environment Variables

```bash
# Optional environment variables
DEVICE=cuda  # or cpu
CONFIDENCE_THRESHOLD=0.5
IMG_SIZE=640
```

## 🚨 Production Considerations

1. **Security**: Configure CORS properly
2. **Scaling**: Use gunicorn with multiple workers
3. **Monitoring**: Add logging and metrics
4. **Rate limiting**: Implement request rate limits
5. **Model optimization**: Consider using TensorRT or ONNX for faster inference

## 📊 Performance

- **Local CPU**: ~100-200ms per image
- **Local GPU**: ~20-50ms per image
- **Cloud deployment**: Varies by service and instance type

## 🐛 Troubleshooting

### Common Issues:

1. **Model not loading**: Check file paths in `MODEL_CONFIG`
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Import errors**: Ensure all dependencies are installed
4. **Docker build fails**: Check Docker daemon is running

### Logs:
Check application logs for detailed error information.
