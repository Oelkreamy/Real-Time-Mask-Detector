import requests

YOLOV8_MODELS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
    "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
    "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
    "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
}

# Replace v0.0.0 with the actual latest version if needed, e.g. v8.0.0

def download_yolov8_models(save_dir="model/weights/yolov8_official"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    for name, url in YOLOV8_MODELS.items():
        out_path = os.path.join(save_dir, name)
        print(f"Downloading {name}...")
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(out_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved to {out_path}")
        else:
            print(f"Failed to download {name} from {url}")

if __name__ == "__main__":
    download_yolov8_models()
