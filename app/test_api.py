"""
Test client for the Mask Detection API
Use this to test your API endpoints
"""

import requests
import json
import base64
from PIL import Image
import io

# API configuration
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_predict_json(image_path: str):
    """Test the /predict endpoint (returns JSON only)"""
    print(f"\n🔍 Testing /predict endpoint with {image_path}...")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            response = requests.post(f"{API_BASE_URL}/predict", files=files)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"Total detections: {result['summary']['total_detections']}")
            print(f"Class counts: {result['summary']['class_counts']}")
            print(f"Compliance rate: {result['summary']['compliance_rate']:.1f}%")
            
            for i, detection in enumerate(result['detections']):
                print(f"  Detection {i+1}: {detection['class_name']} ({detection['confidence']:.2f})")
        else:
            print(f"❌ Prediction failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing predict: {e}")

def test_predict_with_image(image_path: str, save_result: str = "result_annotated.png"):
    """Test the /predict_image endpoint (returns JSON + annotated image)"""
    print(f"\n🖼️  Testing /predict_image endpoint with {image_path}...")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            response = requests.post(f"{API_BASE_URL}/predict_image", files=files)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction with image successful!")
            print(f"Total detections: {result['summary']['total_detections']}")
            print(f"Compliance rate: {result['summary']['compliance_rate']:.1f}%")
            
            # Decode and save the annotated image
            img_data = base64.b64decode(result['annotated_image'])
            img = Image.open(io.BytesIO(img_data))
            img.save(save_result)
            print(f"💾 Annotated image saved as: {save_result}")
            
        else:
            print(f"❌ Prediction failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing predict_image: {e}")

def test_predict_stream(image_path: str, save_result: str = "result_stream.png"):
    """Test the /predict_stream endpoint (returns image directly)"""
    print(f"\n📸 Testing /predict_stream endpoint with {image_path}...")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            response = requests.post(f"{API_BASE_URL}/predict_stream", files=files)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Stream prediction successful!")
            
            # Save the returned image
            with open(save_result, 'wb') as f:
                f.write(response.content)
            print(f"💾 Stream result saved as: {save_result}")
            
        else:
            print(f"❌ Stream prediction failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing predict_stream: {e}")

def main():
    """Run all tests"""
    print("🧪 Starting API Tests...")
    print("=" * 50)
    
    # Test health first
    if not test_health():
        print("❌ API is not healthy, stopping tests")
        return
    
    # You need to provide a test image path
    # Update this path to point to one of your test images
    test_image_path = "path_to_your_test_image.jpg"  # UPDATE THIS PATH
    
    # Check if test image exists
    import os
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        print("Please update the test_image_path in the script")
        return
    
    # Run all endpoint tests
    test_predict_json(test_image_path)
    test_predict_with_image(test_image_path)
    test_predict_stream(test_image_path)
    
    print("\n✅ All tests completed!")
    print("Check the saved result images to verify the detections work correctly.")

if __name__ == "__main__":
    main()
