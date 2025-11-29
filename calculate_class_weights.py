"""
Script to calculate class weights for handling class imbalance in YOLO detection.
This script analyzes your dataset and computes optimal class weights.
"""

import os
import yaml
import torch
import numpy as np
from collections import Counter
from pathlib import Path

def load_dataset_config(config_path):
    """Load dataset configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def count_classes_in_labels(labels_dir, num_classes):
    """Count occurrences of each class in label files"""
    class_counts = Counter()
    total_objects = 0
    
    label_files = list(Path(labels_dir).glob('*.txt'))
    print(f"Found {len(label_files)} label files")
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) >= 5:  # class_id x y w h
                    class_id = int(parts[0])
                    if 0 <= class_id < num_classes:
                        class_counts[class_id] += 1
                        total_objects += 1
    
    return class_counts, total_objects

def calculate_class_weights(class_counts, num_classes, method='inverse_freq'):
    """
    Calculate class weights using different methods:
    
    Methods:
    - 'inverse_freq': 1 / frequency
    - 'balanced': total_samples / (num_classes * class_frequency)
    - 'effective_num': Based on effective number of samples
    """
    
    # Ensure all classes are represented
    for i in range(num_classes):
        if i not in class_counts:
            class_counts[i] = 1  # Assign small count to missing classes
    
    total_samples = sum(class_counts.values())
    weights = {}
    
    if method == 'inverse_freq':
        # Simple inverse frequency
        for class_id in range(num_classes):
            frequency = class_counts[class_id] / total_samples
            weights[class_id] = 1.0 / frequency
            
    elif method == 'balanced':
        # Sklearn-style balanced weights
        for class_id in range(num_classes):
            weights[class_id] = total_samples / (num_classes * class_counts[class_id])
            
    elif method == 'effective_num':
        # Effective number of samples method
        beta = 0.9999  # Hyperparameter
        for class_id in range(num_classes):
            effective_num = (1.0 - np.power(beta, class_counts[class_id])) / (1.0 - beta)
            weights[class_id] = 1.0 / effective_num
    
    # Normalize weights so minimum weight is 1.0
    min_weight = min(weights.values())
    weights = {k: v / min_weight for k, v in weights.items()}
    
    return weights

def analyze_dataset_and_calculate_weights():
    """Main function to analyze dataset and calculate weights"""
    
    # Load dataset configuration
    config_path = "model/config/datasets/mask.yaml"
    
    if not os.path.exists(config_path):
        print(f"Dataset config not found at {config_path}")
        print("Please check the path or create the config file")
        return
    
    config = load_dataset_config(config_path)
    
    # Get dataset information
    train_path = config.get('train', '')
    num_classes = config.get('nc', 3)
    class_names = config.get('names', ['no_mask', 'mask', 'incorrect_mask'])
    
    print(f"Dataset: {config_path}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print("-" * 50)
    
    # Find labels directory
    if os.path.exists(train_path):
        # Assume labels are in train_path/../labels or train_path/labels
        possible_label_dirs = [
            os.path.join(os.path.dirname(train_path), 'labels'),
            os.path.join(train_path, 'labels'),
            train_path.replace('images', 'labels')
        ]
    else:
        possible_label_dirs = [
            'face_mask_dataset6/train/labels',
            'datasets/labels',
            'face_mask_dataset/labels',
            'My First Project.v2i.yolov8/train/labels'
        ]
    
    labels_dir = None
    for dir_path in possible_label_dirs:
        if os.path.exists(dir_path):
            labels_dir = dir_path
            break
    
    if not labels_dir:
        print("Could not find labels directory. Please specify manually:")
        print("Possible locations checked:", possible_label_dirs)
        return
    
    print(f"Using labels directory: {labels_dir}")
    
    # Count classes
    class_counts, total_objects = count_classes_in_labels(labels_dir, num_classes)
    
    print(f"\nClass Distribution:")
    print(f"Total objects: {total_objects}")
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        class_name = class_names[i] if i < len(class_names) else f"class_{i}"
        print(f"  {class_name} (class {i}): {count} objects ({percentage:.1f}%)")
    
    # Calculate weights using different methods
    methods = ['inverse_freq', 'balanced', 'effective_num']
    
    print(f"\nCalculated Class Weights:")
    print("-" * 50)
    
    all_weights = {}
    for method in methods:
        weights = calculate_class_weights(class_counts, num_classes, method)
        all_weights[method] = weights
        
        print(f"\n{method.upper()} method:")
        weight_tensor = []
        for i in range(num_classes):
            class_name = class_names[i] if i < len(class_names) else f"class_{i}"
            weight = weights[i]
            weight_tensor.append(weight)
            print(f"  {class_name} (class {i}): {weight:.3f}")
        
        print(f"  PyTorch tensor: {weight_tensor}")
        print(f"  Code: pos_weight = torch.tensor({weight_tensor})")
    
    # Recommendations
    print(f"\nRecommendations:")
    print("-" * 50)
    
    # Check imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio < 3:
        print("✓ Dataset is relatively balanced. Class weights may not be necessary.")
        recommended_method = None
    elif imbalance_ratio < 10:
        print("⚠ Moderate imbalance detected. Recommend using 'balanced' method.")
        recommended_method = 'balanced'
    else:
        print("🚨 Severe imbalance detected. Recommend using 'effective_num' method.")
        recommended_method = 'effective_num'
    
    if recommended_method:
        weights = all_weights[recommended_method]
        weight_tensor = [weights[i] for i in range(num_classes)]
        print(f"\nRecommended weights: {weight_tensor}")
        
        # Save to file for easy copy-paste
        with open('recommended_class_weights.txt', 'w') as f:
            f.write(f"# Class weights calculated from dataset analysis\n")
            f.write(f"# Method: {recommended_method}\n")
            f.write(f"# Imbalance ratio: {imbalance_ratio:.1f}:1\n\n")
            f.write(f"pos_weight = torch.tensor({weight_tensor})\n")
        
        print("✓ Saved recommended weights to 'recommended_class_weights.txt'")
    
    return all_weights

if __name__ == "__main__":
    analyze_dataset_and_calculate_weights()
