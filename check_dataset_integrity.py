import os
import yaml
from model.data.dataset import Dataset

def check_dataset_integrity(dataset_config, mode='train'):
    print(f"Checking dataset integrity for mode: {mode}")
    # Load config
    config = yaml.safe_load(open(dataset_config, 'r', encoding='utf-8'))
    dataset_path = os.path.join(os.path.dirname(dataset_config), config['path'])
    im_dir = os.path.join(dataset_path, config[mode])
    label_dir = os.path.join(dataset_path, config[mode + '_labels'])

    image_files = sorted([f for f in os.listdir(im_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    label_files = set(os.listdir(label_dir)) if os.path.isdir(label_dir) else set()

    all_ok = True
    for img in image_files:
        img_id = os.path.splitext(img)[0]
        label_file = img_id + '.txt'
        img_path = os.path.join(im_dir, img)
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            print(f"[MISSING LABEL] Image '{img}' has no annotation file '{label_file}'!")
            all_ok = False
            continue
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines) == 0:
            print(f"[EMPTY LABEL] Image '{img}' has an empty annotation file '{label_file}'!")
            all_ok = False
    # Optionally, check for orphan label files (labels with no image)
    for label_file in label_files:
        img_file = os.path.splitext(label_file)[0] + '.jpg'
        if img_file not in image_files:
            print(f"[ORPHAN LABEL] Label file '{label_file}' has no corresponding image!")
            all_ok = False
    if all_ok:
        print("All images and labels are present and non-empty.")
    print("Check completed.")

if __name__ == '__main__':
    # Allow running directly from VS Code Run button by setting defaults here
    DEFAULT_DATASET = 'model/config/datasets/mask.yaml'  # Update this path if needed
    DEFAULT_MODE = 'train'
    try:
        import argparse
        parser = argparse.ArgumentParser(description='Check dataset integrity (images/labels)')
        parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET, help='Path to dataset config YAML')
        parser.add_argument('--mode', type=str, default=DEFAULT_MODE, help='Dataset mode (train/val/test)')
        args = parser.parse_args()
        check_dataset_integrity(args.dataset, mode=args.mode)
    except SystemExit:
        # If run from VS Code Run button (no args), use defaults
        print(f"No command-line arguments provided. Using defaults: dataset={DEFAULT_DATASET}, mode={DEFAULT_MODE}")
        check_dataset_integrity(DEFAULT_DATASET, mode=DEFAULT_MODE)
