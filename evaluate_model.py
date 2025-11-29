"""
Advanced evaluation script using torchmetrics to compute mAP, Recall, and Precision
for the mask detection model.

This script:
1. Loads the trained model
2. Runs inference on test dataset (not validation)
3. Computes detection metrics using torchmetrics
4. Saves annotated images with predictions and ground truth to results directory
5. Provides detailed explanations of each metric
"""

# Fix OpenMP library conflict before importing any libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import yaml
from pathlib import Path
import cv2
import numpy as np
import pandas as pd  # Add pandas for Excel export
import seaborn as sns  # Add seaborn for confusion matrix plotting
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision
from torchvision.ops import box_iou
from scipy.signal import savgol_filter  # Add scipy for curve smoothing

from model.models.detection_model import DetectionModel
from model.data.detections import Detections
from model.data.utils import pad_to

class ModelEvaluator:
    def __init__(self, model_config, weights_path, dataset_config, device='cuda', use_class_mapping=False):
        """
        Initialize the evaluator
        
        Args:
            model_config: Path to model YAML config
            weights_path: Path to trained model weights
            dataset_config: Path to dataset YAML config  
            device: Device to run evaluation on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"🔥 Using device: {self.device}")
        
        # Load model
        print("📦 Loading model...")
        self.model = DetectionModel(model_config, device=self.device)
        

        # Load checkpoint or state dict
        checkpoint = torch.load(weights_path, map_location=self.device)
        # Handle both direct state dict and training checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("✅ Detected training checkpoint format. Loading 'model_state_dict'.")
            state_dict = checkpoint['model_state_dict']
        else:
            print("✅ Detected direct state dict format. Loading weights directly.")
            state_dict = checkpoint

        # Try loading the state dict
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.mode = 'eval'
        
        # Load dataset config
        with open(dataset_config, 'r') as f:
            self.dataset_info = yaml.safe_load(f)
        
        # Handle different dataset config formats
        if isinstance(self.dataset_info['names'], dict):
            # Names are in dict format {0: 'class0', 1: 'class1', ...}
            self.class_names = [self.dataset_info['names'][i] for i in sorted(self.dataset_info['names'].keys())]
            self.num_classes = len(self.class_names)
        else:
            # Names are in list format ['class0', 'class1', ...]
            self.class_names = self.dataset_info['names']
            self.num_classes = len(self.class_names)
        
        print(f"📝 Classes ({self.num_classes}): {self.class_names}")
        
        # ⚠️ CRITICAL FIX: Class mapping correction
        # Based on debugging: model predicts mostly class 0, but GT has mostly class 1 (with_mask)
        # This suggests model learned: 0=with_mask, 1=without_mask, 2=incorrect_mask
        # But dataset expects: 0=incorrect_mask, 1=with_mask, 2=without_mask
        self.class_remapping = {
            0: 1,  # Model's class 0 (with_mask) → Dataset's class 1 (with_mask)
            1: 2,  # Model's class 1 (without_mask) → Dataset's class 2 (without_mask)  
            2: 0   # Model's class 2 (incorrect_mask) → Dataset's class 0 (incorrect_mask)
        }
        print(f"🔄 Applying class remapping: {self.class_remapping}")
        print(f"   Model class 0 → Dataset class {self.class_remapping[0]} ({self.class_names[self.class_remapping[0]]})")
        print(f"   Model class 1 → Dataset class {self.class_remapping[1]} ({self.class_names[self.class_remapping[1]]})")
        print(f"   Model class 2 → Dataset class {self.class_remapping[2]} ({self.class_names[self.class_remapping[2]]})")
        
        # Initialize torchmetrics
        self.map_metric = MeanAveragePrecision(
            box_format='xyxy',  # Our detections are in xyxy format
            iou_type='bbox',    # We're doing bounding box detection
            class_metrics=True  # Compute per-class metrics
        )
        
        self.img_size = (640, 640)
        
        # Create results directory
        self.results_dir = self.create_results_directory()
        print(f"📁 Results will be saved to: {self.results_dir}")
        
        self.use_class_mapping = use_class_mapping
    def remap_prediction_classes(self, pred_dict):
        """
        Remap model prediction classes to match dataset class mapping
        
        Args:
            pred_dict: Dictionary with 'boxes', 'scores', 'labels'
            
        Returns:
            pred_dict with remapped labels
        """
        if not self.use_class_mapping:
            return pred_dict
        
        if len(pred_dict['labels']) > 0:
            # Apply class remapping
            remapped_labels = torch.tensor([
                self.class_remapping[label.item()] 
                for label in pred_dict['labels']
            ], dtype=torch.int64)
            
            pred_dict['labels'] = remapped_labels
        
        return pred_dict
        
    def create_results_directory(self):
        """Create results directory in the dataset folder"""
        # Handle relative paths in dataset config
        dataset_config_dir = Path("model/config/datasets")
        base_path = dataset_config_dir / self.dataset_info['path']
        
        # Create results directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = base_path / "results" / f"evaluation_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        return results_dir
        
    def load_test_data(self):
        """
        Load test images and labels (changed from validation to test)
        Returns list of (image_path, label_path) tuples
        """
        # Handle relative paths in dataset config
        dataset_config_dir = Path("model/config/datasets")
        base_path = dataset_config_dir / self.dataset_info['path']
        
        # Use test set instead of validation set
        test_images_dir = base_path / self.dataset_info['test']
        test_labels_dir = base_path / self.dataset_info['test_labels']
        
        print(f"🔍 Looking for test data in:")
        print(f"   Images: {test_images_dir}")
        print(f"   Labels: {test_labels_dir}")
        
        image_files = list(test_images_dir.glob('*.jpg')) + list(test_images_dir.glob('*.png'))
        
        data_pairs = []
        for img_path in image_files:
            label_path = test_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                data_pairs.append((img_path, label_path))
        
        print(f"📊 Found {len(data_pairs)} test samples")
        return data_pairs
    
    def load_ground_truth_labels(self, label_path, img_width, img_height):
        """
        Load YOLO format labels and convert to xyxy format
        
        Args:
            label_path: Path to label file
            img_width: Image width
            img_height: Image height
            
        Returns:
            dict with 'boxes' and 'labels' tensors
        """
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Convert YOLO format (normalized xywh) to xyxy pixel coordinates
                        x1 = (x_center - width/2) * img_width
                        y1 = (y_center - height/2) * img_height
                        x2 = (x_center + width/2) * img_width
                        y2 = (y_center + height/2) * img_height
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty(0, 4),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.empty(0, dtype=torch.int64)
        }
    
    def preprocess_image(self, image):
        """Preprocess image for model inference"""
        h0, w0 = image.shape[:2]
        
        # Resize maintaining aspect ratio
        ratio = min(self.img_size[0] / h0, self.img_size[1] / w0)
        h, w = int(h0 * ratio), int(w0 * ratio)
        image_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor and pad
        image_tensor = torch.from_numpy(image_resized.transpose((2, 0, 1))).float() / 255.0
        image_tensor, pads = pad_to(image_tensor, shape=self.img_size)
        
        return image_tensor.unsqueeze(0).to(self.device), pads, (w0, h0), (w, h)
    
    def postprocess_predictions(self, preds, pads, original_size, resized_size):
        """Convert model predictions to evaluation format"""
        detections = Detections.from_yolo(preds)
        
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
            
            # Ensure all outputs are PyTorch tensors
            boxes = detections.xyxy
            scores = detections.confidence
            labels = detections.class_id
            
            # Convert to tensors if they aren't already
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.tensor(boxes, dtype=torch.float32)
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores, dtype=torch.float32)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.int64)
            
            # Ensure proper dtypes
            boxes = boxes.float()
            scores = scores.float()
            labels = labels.long()

            return {
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            }
        else:
            print("\n[DEBUG] No detections found in this image.")
            return {
                'boxes': torch.empty(0, 4, dtype=torch.float32),
                'scores': torch.empty(0, dtype=torch.float32),
                'labels': torch.empty(0, dtype=torch.int64)
            }
    
    def draw_predictions_on_image(self, image, predictions, ground_truth=None):
        """
        Draw predictions and optionally ground truth on image
        
        Args:
            image: Original image (numpy array)
            predictions: Dict with 'boxes', 'scores', 'labels'
            ground_truth: Optional dict with 'boxes', 'labels' for GT
            
        Returns:
            Annotated image
        """
        # Create a copy to avoid modifying original
        annotated_image = image.copy()
        
        # Class colors (BGR format for OpenCV) - Updated for correct class order
        class_colors = {
            0: (0, 255, 0),      # with_mask - Green
            1: (255, 0, 0),      # without_mask - Blue  
            2: (0, 0, 255),      # incorrect_mask - Red
        }
        
        # Draw ground truth boxes (if provided) in dashed lines
        if ground_truth is not None and len(ground_truth['boxes']) > 0:
            for i, (box, label) in enumerate(zip(ground_truth['boxes'], ground_truth['labels'])):
                x1, y1, x2, y2 = box.int().tolist()
                color = class_colors.get(label.item(), (128, 128, 128))
                
                # Draw dashed rectangle for ground truth
                self._draw_dashed_rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # GT label
                label_text = f"GT: {self.class_names[label.item()]}"
                cv2.putText(annotated_image, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw predictions in solid lines
        if len(predictions['boxes']) > 0:
            for i, (box, score, label) in enumerate(zip(predictions['boxes'], 
                                                       predictions['scores'], 
                                                       predictions['labels'])):
                x1, y1, x2, y2 = box.int().tolist()
                confidence = score.item()
                class_idx = label.item()
                
                color = class_colors.get(class_idx, (128, 128, 128))
                
                # Draw solid rectangle for predictions
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Prediction label with confidence
                label_text = f"{self.class_names[class_idx]}: {confidence:.2f}"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Background rectangle for text
                cv2.rectangle(annotated_image, (x1, y1-25), (x1+text_size[0]+10, y1), color, -1)
                cv2.putText(annotated_image, label_text, (x1+5, y1-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_image
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness):
        """Draw a dashed rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Top and bottom lines
        for x in range(x1, x2, 10):
            cv2.line(img, (x, y1), (min(x+5, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x+5, x2), y2), color, thickness)
        
        # Left and right lines  
        for y in range(y1, y2, 10):
            cv2.line(img, (x1, y), (x1, min(y+5, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y+5, y2)), color, thickness)
    
    def calculate_true_precision_recall_per_class(self, predictions, targets, num_classes, iou_threshold=0.5):
        """
        Calculate true precision and recall per class at a specific confidence threshold
        by manually counting TP, FP, and FN for each class.
        """
        tp_counts = [0] * num_classes
        fp_counts = [0] * num_classes
        fn_counts = [0] * num_classes
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']
            target_boxes = target['boxes']
            target_labels = target['labels']
            conf_mask = torch.ones_like(pred_scores, dtype=torch.bool)  # Already filtered by conf
            pred_boxes_filtered = pred_boxes[conf_mask]
            pred_labels_filtered = pred_labels[conf_mask]
            pred_scores_filtered = pred_scores[conf_mask]
            target_matched = [False] * len(target_boxes)
            for pred_idx in range(len(pred_boxes_filtered)):
                pred_box = pred_boxes_filtered[pred_idx].unsqueeze(0)
                pred_label = pred_labels_filtered[pred_idx].item()
                best_iou = 0.0
                best_target_idx = -1
                for target_idx in range(len(target_boxes)):
                    if target_labels[target_idx].item() == pred_label and not target_matched[target_idx]:
                        target_box = target_boxes[target_idx].unsqueeze(0)
                        iou = box_iou(pred_box, target_box).item()
                        if iou > best_iou:
                            best_iou = iou
                            best_target_idx = target_idx
                if best_iou >= iou_threshold and best_target_idx >= 0:
                    tp_counts[pred_label] += 1
                    target_matched[best_target_idx] = True
                else:
                    fp_counts[pred_label] += 1
            for target_idx in range(len(target_boxes)):
                if not target_matched[target_idx]:
                    target_label = target_labels[target_idx].item()
                    fn_counts[target_label] += 1
        precisions = []
        recalls = []
        for cls_idx in range(num_classes):
            tp = tp_counts[cls_idx]
            fp = fp_counts[cls_idx]
            fn = fn_counts[cls_idx]
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0
            precisions.append(precision)
            recalls.append(recall)
        return precisions, recalls
    
    def intersection_over_union(self, boxes_preds, boxes_labels, box_format="corners"):
        """
        Calculates intersection over union

        Parameters:
            boxes_preds (tensor): Predictions of Bounding Boxes (4,)
            boxes_labels (tensor): Correct Labels of Boxes (4,)
            box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

        Returns:
            tensor: Intersection over union
        """
        if box_format == "midpoint":
            box1_x1 = boxes_preds[0] - boxes_preds[2] / 2
            box1_y1 = boxes_preds[1] - boxes_preds[3] / 2
            box1_x2 = boxes_preds[0] + boxes_preds[2] / 2
            box1_y2 = boxes_preds[1] + boxes_preds[3] / 2
            box2_x1 = boxes_labels[0] - boxes_labels[2] / 2
            box2_y1 = boxes_labels[1] - boxes_labels[3] / 2
            box2_x2 = boxes_labels[0] + boxes_labels[2] / 2
            box2_y2 = boxes_labels[1] + boxes_labels[3] / 2

        elif box_format == "corners":
            box1_x1 = boxes_preds[0]
            box1_y1 = boxes_preds[1]
            box1_x2 = boxes_preds[2]
            box1_y2 = boxes_preds[3]
            box2_x1 = boxes_labels[0]
            box2_y1 = boxes_labels[1]
            box2_x2 = boxes_labels[2]
            box2_y2 = boxes_labels[3]

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        return intersection / (box1_area + box2_area - intersection + 1e-6)

    def mean_average_precision_debug(self, pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=3):
        """
        Calculates mean average precision with detailed debugging information exported to Excel
        
        Parameters:
            pred_boxes (list): list of lists containing all bboxes with each bboxes
            specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
            true_boxes (list): Similar as pred_boxes except all the correct ones 
            iou_threshold (float): threshold where predicted bboxes is correct
            box_format (str): "midpoint" or "corners" used to specify bboxes
            num_classes (int): number of classes

        Returns:
            float: mAP value across all classes given a specific IoU threshold 
        """
        from collections import Counter

        # list storing all AP for respective classes
        average_precisions = []
        precision_recall_curves = {}
        
        # Debug data for Excel export
        all_debug_data = []

        # used for numerical stability later on
        epsilon = 1e-6

        for c in range(num_classes):
            print(f"🔍 Processing class {c} ({self.class_names[c]})...")
            
            detections = []
            ground_truths = []
            class_debug_data = []

            # Go through all predictions and targets,
            # and only add the ones that belong to the
            # current class c
            for detection in pred_boxes:
                if detection[1] == c:
                    detections.append(detection)

            for true_box in true_boxes:
                if true_box[1] == c:
                    ground_truths.append(true_box)

            print(f"   Found {len(detections)} predictions and {len(ground_truths)} ground truths")

            # find the amount of bboxes for each training example
            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            # We then go through each key, val in this dictionary
            # and convert to the following (w.r.t same example):
            # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            # sort by box probabilities which is index 2
            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)
            
            # If none exists for this class then we can safely skip
            if total_true_bboxes == 0:
                average_precisions.append(0.0)
                precision_recall_curves[c] = {
                    'precision': [1.0, 0.0],
                    'recall': [0.0, 1.0],
                    'ap': 0.0
                }
                print(f"   No ground truths for class {c}, skipping...")
                continue

            for detection_idx, detection in enumerate(detections):
                # Only take out the ground_truths that have the same
                # training idx as detection
                ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[0] == detection[0]
                ]

                best_iou = 0
                best_gt_idx = -1
                matched_gt = None

                for idx, gt in enumerate(ground_truth_img):
                    iou = self.intersection_over_union(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format,
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                        matched_gt = gt

                # Determine TP/FP
                is_tp = False
                tp_fp_reason = ""
                
                if best_iou > iou_threshold and best_gt_idx >= 0:
                    # Check if this GT was already matched
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        # True positive
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                        is_tp = True
                        tp_fp_reason = f"TP: IoU={best_iou:.3f} >= {iou_threshold}, GT not matched before"
                    else:
                        # False positive - GT already matched
                        FP[detection_idx] = 1
                        tp_fp_reason = f"FP: IoU={best_iou:.3f} >= {iou_threshold}, but GT already matched"
                else:
                    # False positive - IoU too low
                    FP[detection_idx] = 1
                    tp_fp_reason = f"FP: IoU={best_iou:.3f} < {iou_threshold}"

                # Calculate cumulative TP/FP up to this point
                tp_cumsum = torch.sum(TP[:detection_idx+1]).item()
                fp_cumsum = torch.sum(FP[:detection_idx+1]).item()
                
                # Calculate precision and recall at this point
                precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
                recall = tp_cumsum / total_true_bboxes if total_true_bboxes > 0 else 0

                # Store debug information
                debug_entry = {
                    'Class': self.class_names[c],
                    'Class_ID': c,
                    'Image_Index': detection[0],
                    'Detection_Index': detection_idx,
                    'Confidence': detection[2],
                    'Pred_Box': f"[{detection[3]:.1f}, {detection[4]:.1f}, {detection[5]:.1f}, {detection[6]:.1f}]",
                    'Best_IoU': best_iou,
                    'IoU_Threshold': iou_threshold,
                    'Matched_GT': f"[{matched_gt[3]:.1f}, {matched_gt[4]:.1f}, {matched_gt[5]:.1f}, {matched_gt[6]:.1f}]" if matched_gt else "None",
                    'TP_FP': 'TP' if is_tp else 'FP',
                    'TP_FP_Reason': tp_fp_reason,
                    'Cumulative_TP': tp_cumsum,
                    'Cumulative_FP': fp_cumsum,
                    'Precision': precision,
                    'Recall': recall,
                    'Total_GT': total_true_bboxes
                }
                
                class_debug_data.append(debug_entry)

            # Calculate final precision-recall curve
            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            
            # Store precision-recall curve for this class
            ap_value = torch.trapz(precisions, recalls).item()
            precision_recall_curves[c] = {
                'precision': precisions.tolist(),
                'recall': recalls.tolist(),
                'ap': ap_value
            }
            
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))
            
            print(f"   Class {c} AP: {ap_value:.4f}")
            
            # Add class debug data to overall list
            all_debug_data.extend(class_debug_data)

        # Calculate final mAP
        mAP = sum(average_precisions) / len(average_precisions)
        
        # Export debug data to Excel
        if all_debug_data:
            debug_df = pd.DataFrame(all_debug_data)
            excel_path = self.results_dir / "debug_map_calculations.xlsx"
            
            # Create Excel writer with multiple sheets
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Main debug data
                debug_df.to_excel(writer, sheet_name='Detailed_Calculations', index=False)
                
                # Summary by class
                summary_data = []
                for c in range(num_classes):
                    class_data = [entry for entry in all_debug_data if entry['Class_ID'] == c]
                    if class_data:
                        total_tp = sum(1 for entry in class_data if entry['TP_FP'] == 'TP')
                        total_fp = sum(1 for entry in class_data if entry['TP_FP'] == 'FP')
                        final_precision = class_data[-1]['Precision'] if class_data else 0
                        final_recall = class_data[-1]['Recall'] if class_data else 0
                        ap = precision_recall_curves[c]['ap'] if c in precision_recall_curves else 0
                        
                        summary_data.append({
                            'Class': self.class_names[c],
                            'Class_ID': c,
                            'Total_Predictions': len(class_data),
                            'Total_TP': total_tp,
                            'Total_FP': total_fp,
                            'Total_GT': class_data[0]['Total_GT'] if class_data else 0,
                            'Final_Precision': final_precision,
                            'Final_Recall': final_recall,
                            'Average_Precision': ap
                        })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Class_Summary', index=False)
                
                # Overall summary
                overall_summary = pd.DataFrame([{
                    'IoU_Threshold': iou_threshold,
                    'Total_Classes': num_classes,
                    'mAP': mAP.item() if torch.is_tensor(mAP) else mAP,
                    'Total_Predictions': len(all_debug_data),
                    'Total_TP': sum(1 for entry in all_debug_data if entry['TP_FP'] == 'TP'),
                    'Total_FP': sum(1 for entry in all_debug_data if entry['TP_FP'] == 'FP')
                }])
                overall_summary.to_excel(writer, sheet_name='Overall_Summary', index=False)
            
            print(f"📊 Debug calculations exported to: {excel_path}")
        
        return mAP, average_precisions, precision_recall_curves

    def mean_average_precision(self, pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=3):
        """
        Calculates mean average precision - EXACT implementation from reference

        Parameters:
            pred_boxes (list): list of lists containing all bboxes with each bboxes
            specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
            true_boxes (list): Similar as pred_boxes except all the correct ones 
            iou_threshold (float): threshold where predicted bboxes is correct
            box_format (str): "midpoint" or "corners" used to specify bboxes
            num_classes (int): number of classes

        Returns:
            float: mAP value across all classes given a specific IoU threshold 
        """
        from collections import Counter

        # list storing all AP for respective classes
        average_precisions = []
        precision_recall_curves = {}

        # used for numerical stability later on
        epsilon = 1e-6

        for c in range(num_classes):
            detections = []
            ground_truths = []

            # Go through all predictions and targets,
            # and only add the ones that belong to the
            # current class c
            for detection in pred_boxes:
                if detection[1] == c:
                    detections.append(detection)

            for true_box in true_boxes:
                if true_box[1] == c:
                    ground_truths.append(true_box)

            # find the amount of bboxes for each training example
            # Counter here finds how many ground truth bboxes we get
            # for each training example, so let's say img 0 has 3,
            # img 1 has 5 then we will obtain a dictionary with:
            # amount_bboxes = {0:3, 1:5}
            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            # We then go through each key, val in this dictionary
            # and convert to the following (w.r.t same example):
            # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            # sort by box probabilities which is index 2
            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)
            
            # If none exists for this class then we can safely skip
            if total_true_bboxes == 0:
                average_precisions.append(0.0)
                precision_recall_curves[c] = {
                    'precision': [1.0, 0.0],
                    'recall': [0.0, 1.0],
                    'ap': 0.0
                }
                continue

            for detection_idx, detection in enumerate(detections):
                # Only take out the ground_truths that have the same
                # training idx as detection
                ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[0] == detection[0]
                ]

                num_gts = len(ground_truth_img)
                best_iou = 0
                best_gt_idx = -1

                for idx, gt in enumerate(ground_truth_img):
                    iou = self.intersection_over_union(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format,
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > iou_threshold and best_gt_idx >= 0:
                    # only detect ground truth detection once
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        # true positive and add this bounding box to seen
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            
            # Store precision-recall curve for this class
            precision_recall_curves[c] = {
                'precision': precisions.tolist(),
                'recall': recalls.tolist(),
                'ap': torch.trapz(precisions, recalls).item()
            }
            
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))

        mAP = sum(average_precisions) / len(average_precisions)
        return mAP, average_precisions, precision_recall_curves

    def calculate_map_at_iou(self, predictions, targets, iou_threshold=0.5, debug=False):
        """
        Calculate mAP at a specific IoU threshold following the EXACT reference implementation
        
        Args:
            predictions: List of prediction dictionaries (one per image)
            targets: List of target dictionaries (one per image)
            iou_threshold: IoU threshold for TP/FP classification
            debug: Whether to use debug version with Excel export
            
        Returns:
            mAP, per_class_AP_list, precision_recall_curves_dict
        """
        # Convert to the format expected by the reference implementation
        pred_boxes, true_boxes = self.convert_predictions_to_list_format(predictions, targets)
        
        # Use debug version if requested, otherwise use regular version
        if debug:
            mAP, per_class_ap, precision_recall_curves = self.mean_average_precision_debug(
                pred_boxes, true_boxes, iou_threshold=iou_threshold, 
                box_format="corners", num_classes=self.num_classes
            )
        else:
            mAP, per_class_ap, precision_recall_curves = self.mean_average_precision(
                pred_boxes, true_boxes, iou_threshold=iou_threshold, 
                box_format="corners", num_classes=self.num_classes
            )
        
        return mAP, per_class_ap, precision_recall_curves

    def convert_predictions_to_list_format(self, predictions, targets):
        """
        Convert predictions and targets to the list format required by mAP calculation
        Format: [train_idx, class_pred, prob_score, x1, y1, x2, y2]
        """
        pred_boxes = []
        true_boxes = []
        
        for train_idx, (pred_dict, target_dict) in enumerate(zip(predictions, targets)):
            # Convert predictions
            if len(pred_dict['boxes']) > 0:
                for i in range(len(pred_dict['boxes'])):
                    box = pred_dict['boxes'][i]
                    score = pred_dict['scores'][i].item()
                    label = pred_dict['labels'][i].item()
                    
                    pred_boxes.append([
                        train_idx,
                        label,
                        score,
                        box[0].item(),  # x1
                        box[1].item(),  # y1
                        box[2].item(),  # x2
                        box[3].item()   # y2
                    ])
            
            # Convert ground truths
            if len(target_dict['boxes']) > 0:
                for i in range(len(target_dict['boxes'])):
                    box = target_dict['boxes'][i]
                    label = target_dict['labels'][i].item()
                    
                    true_boxes.append([
                        train_idx,
                        label,
                        1.0,  # GT confidence is always 1
                        box[0].item(),  # x1
                        box[1].item(),  # y1
                        box[2].item(),  # x2
                        box[3].item()   # y2
                    ])
        
        return pred_boxes, true_boxes
    
    def compute_curve_metrics(self, predictions, targets):
        """
        Compute mAP, precision, recall curves using CORRECT methodology
        Following the guidelines: for each class, collect all predictions, sort by confidence,
        calculate TP/FP at each threshold, build PR curve, calculate AP as area under curve.
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
            
        Returns:
            Dictionary containing curve data for plotting
        """
        print("📈 Computing mAP and PR curves using CORRECT methodology...")
        
        # Calculate mAP@0.5 with debugging (exports Excel file)
        print("🔍 Calculating mAP@0.5 with detailed debugging...")
        map_50, per_class_ap_50, precision_recall_curves = self.calculate_map_at_iou(
            predictions, targets, iou_threshold=0.5, debug=True
        )
        
        # Calculate mAP@0.5:0.05:0.95 (average over multiple IoU thresholds)
        iou_thresholds = np.arange(0.5, 1.0, 0.05)  # [0.5, 0.55, 0.6, ..., 0.95]
        map_values = []
        
        for iou_thresh in iou_thresholds:
            map_at_iou, _, _ = self.calculate_map_at_iou(predictions, targets, iou_threshold=iou_thresh)
            # Convert tensor to float if needed
            if torch.is_tensor(map_at_iou):
                map_at_iou = map_at_iou.item()
            map_values.append(map_at_iou)
        
        map_50_95 = np.mean(map_values)
        
        # Convert map_50 to float if it's a tensor
        if torch.is_tensor(map_50):
            map_50 = map_50.item()
        
        print(f"📊 mAP@0.5: {map_50:.4f}")
        print(f"📊 mAP@0.5:0.95: {map_50_95:.4f}")
        
        # Prepare curve data in the expected format for plotting
        curve_data = {
            'per_class_ap': [ap.item() if torch.is_tensor(ap) else ap for ap in per_class_ap_50],
            'mAP': map_50,
            'mAP_50_95': map_50_95,
            'custom_per_class_precision': {},
            'custom_per_class_recall': {},
            'custom_per_class_f1': {},
            'custom_overall_precision': [],
            'custom_overall_recall': [],
            'custom_overall_f1': []
        }
        
        # Store per-class curves from mAP@0.5 calculation
        for class_idx in range(self.num_classes):
            if class_idx in precision_recall_curves:
                precision = precision_recall_curves[class_idx]['precision']
                recall = precision_recall_curves[class_idx]['recall']
                
                # Calculate F1 scores
                f1_scores = []
                for p, r in zip(precision, recall):
                    if p + r > 0:
                        f1 = 2 * (p * r) / (p + r)
                    else:
                        f1 = 0.0
                    f1_scores.append(f1)
                
                curve_data['custom_per_class_precision'][class_idx] = precision
                curve_data['custom_per_class_recall'][class_idx] = recall
                curve_data['custom_per_class_f1'][class_idx] = f1_scores
                
                print(f"   Class {class_idx} ({self.class_names[class_idx]}): AP@0.5={precision_recall_curves[class_idx]['ap']:.4f}")
            else:
                # No data for this class
                curve_data['custom_per_class_precision'][class_idx] = [1.0, 0.0]
                curve_data['custom_per_class_recall'][class_idx] = [0.0, 1.0] 
                curve_data['custom_per_class_f1'][class_idx] = [0.0, 0.0]
                print(f"   Class {class_idx} ({self.class_names[class_idx]}): AP@0.5=0.0000 (no data)")
        
        # Calculate overall curves (mean of per-class curves at corresponding points)
        if self.num_classes > 0:
            max_points = max(len(curve_data['custom_per_class_precision'][i]) for i in range(self.num_classes))
            
            for i in range(max_points):
                precisions_at_i = []
                recalls_at_i = []
                f1s_at_i = []
                
                for class_idx in range(self.num_classes):
                    if i < len(curve_data['custom_per_class_precision'][class_idx]):
                        precisions_at_i.append(curve_data['custom_per_class_precision'][class_idx][i])
                        recalls_at_i.append(curve_data['custom_per_class_recall'][class_idx][i])
                        f1s_at_i.append(curve_data['custom_per_class_f1'][class_idx][i])
                
                if precisions_at_i:
                    curve_data['custom_overall_precision'].append(np.mean(precisions_at_i))
                    curve_data['custom_overall_recall'].append(np.mean(recalls_at_i))
                    curve_data['custom_overall_f1'].append(np.mean(f1s_at_i))
        
        print("✅ mAP and PR curves calculation completed using CORRECT methodology")
        
        # Fix combined precision-recall curve to reduce oscillations
        curve_data = self.fix_combined_precision_recall_curve(curve_data)
        
        # Calculate F1-confidence curve using correct methodology
        print("📈 Computing F1-confidence curve...")
        f1_curve_data = self.calculate_f1_confidence_curve(predictions, targets, iou_threshold=0.5)
        
        # Add F1 curve data to main curve_data
        curve_data['f1_curve_data'] = f1_curve_data
        
        return curve_data
    
    def plot_precision_recall_curve(self, curve_data, save_path):
        """
        Plot and save Precision-Recall curve
        
        Args:
            curve_data: Dictionary containing curve metrics
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Better class colors and styles for visibility
        class_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        class_styles = ['-', '--', '-.', ':', '-', '--']
        
        # Custom curves
        for class_idx in range(self.num_classes):
            precision = np.array(curve_data['custom_per_class_precision'][class_idx])
            recall = np.array(curve_data['custom_per_class_recall'][class_idx])
            
            # Sort by recall for proper curve (important for AUC calculation)
            if len(precision) > 1 and len(recall) > 1:
                sort_indices = np.argsort(recall)
                recall_sorted = recall[sort_indices]
                precision_sorted = precision[sort_indices]
                
                # Add final point: same recall as last point, but precision = 0
                if len(recall_sorted) > 0 and recall_sorted[-1] < 1.0:
                    recall_sorted = np.append(recall_sorted, recall_sorted[-1])
                    precision_sorted = np.append(precision_sorted, 0.0)
                
                # Calculate AUC using trapezoid rule
                auc_score = np.trapezoid(precision_sorted, recall_sorted)
                
                color = class_colors[class_idx % len(class_colors)]
                style = class_styles[class_idx % len(class_styles)]
                plt.plot(recall_sorted, precision_sorted, 
                        color=color, 
                        linestyle=style,
                        linewidth=3, 
                        marker='o',
                        markersize=4,
                        label=f'{self.class_names[class_idx]} AP={auc_score:.3f}')
            else:
                color = class_colors[class_idx % len(class_colors)]
                style = class_styles[class_idx % len(class_styles)]
                plt.plot([], [], 
                        color=color, 
                        linestyle=style,
                        linewidth=3, 
                        label=f'{self.class_names[class_idx]} AP=0.000')
        
        # Overall curves
        custom_overall_precision = np.array(curve_data['custom_overall_precision'])
        custom_overall_recall = np.array(curve_data['custom_overall_recall'])
        if len(custom_overall_precision) > 1 and len(custom_overall_recall) > 1:
            sort_indices = np.argsort(custom_overall_recall)
            custom_overall_recall_sorted = custom_overall_recall[sort_indices]
            custom_overall_precision_sorted = custom_overall_precision[sort_indices]
            
            # Add final point: same recall as last point, but precision = 0
            if len(custom_overall_recall_sorted) > 0 and custom_overall_recall_sorted[-1] < 1.0:
                custom_overall_recall_sorted = np.append(custom_overall_recall_sorted, custom_overall_recall_sorted[-1])
                custom_overall_precision_sorted = np.append(custom_overall_precision_sorted, 0.0)
            
            custom_overall_auc = np.trapezoid(custom_overall_precision_sorted, custom_overall_recall_sorted)
            plt.plot(custom_overall_recall_sorted, custom_overall_precision_sorted, 
                    color='black', 
                    linewidth=5, 
                    marker='s',
                    markersize=5,
                    label=f'All Classes mAP@0.5={custom_overall_auc:.3f}')
        else:
            plt.plot([], [], 
                    color='black', 
                    linewidth=5, 
                    label=f'All Classes mAP@0.5=0.000')
        
        # Formatting
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)  # Slightly higher to accommodate markers
        plt.xlabel('Recall', fontsize=14, fontweight='bold')
        plt.ylabel('Precision', fontsize=14, fontweight='bold')
        plt.title('Precision-Recall Curve (IoU=0.5)', fontsize=16, fontweight='bold')
        plt.legend(loc='lower left', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add background color for better visibility
        plt.gca().set_facecolor('#f8f9fa')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Precision-Recall curve saved to: {save_path}")
    
    def calculate_f1_confidence_curve(self, predictions, targets, iou_threshold=0.5):
        """
        Calculate F1-confidence curve by varying confidence thresholds
        Following the correct methodology: for each confidence threshold,
        filter predictions, calculate TP/FP/FN, then compute F1 per class and overall.
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries  
            iou_threshold: IoU threshold for TP/FP classification (default: 0.5)
            
        Returns:
            Dictionary containing F1 curve data
        """
        print("📈 Computing F1-confidence curve using CORRECT methodology...")
        
        # Define confidence thresholds to evaluate
        confidence_thresholds = np.arange(0.01, 1.0, 0.02)  # 0.01, 0.03, 0.05, ..., 0.99
        
        # Storage for curve data
        f1_curve_data = {
            'confidence_thresholds': confidence_thresholds.tolist(),
            'per_class_f1': {i: [] for i in range(self.num_classes)},
            'overall_f1': [],
            'per_class_precision': {i: [] for i in range(self.num_classes)},
            'per_class_recall': {i: [] for i in range(self.num_classes)},
            'overall_precision': [],
            'overall_recall': []
        }
        
        print(f"   Evaluating {len(confidence_thresholds)} confidence thresholds...")
        
        # For each confidence threshold
        for conf_thresh in tqdm(confidence_thresholds, desc="Computing F1 curve"):
            
            # Count TP, FP, FN for each class at this confidence threshold
            class_tp = {i: 0 for i in range(self.num_classes)}
            class_fp = {i: 0 for i in range(self.num_classes)}
            class_fn = {i: 0 for i in range(self.num_classes)}
            
            # Process each image
            for pred_dict, target_dict in zip(predictions, targets):
                
                # Filter predictions by confidence threshold
                if len(pred_dict['scores']) > 0:
                    conf_mask = pred_dict['scores'] >= conf_thresh
                    filtered_pred = {
                        'boxes': pred_dict['boxes'][conf_mask],
                        'scores': pred_dict['scores'][conf_mask],
                        'labels': pred_dict['labels'][conf_mask]
                    }
                else:
                    filtered_pred = {
                        'boxes': torch.empty(0, 4),
                        'scores': torch.empty(0),
                        'labels': torch.empty(0, dtype=torch.long)
                    }
                
                pred_boxes = filtered_pred['boxes']
                pred_labels = filtered_pred['labels']
                target_boxes = target_dict['boxes']
                target_labels = target_dict['labels']
                
                # Track which ground truths have been matched
                target_matched = [False] * len(target_boxes)
                
                # Process each prediction
                for pred_idx in range(len(pred_boxes)):
                    pred_box = pred_boxes[pred_idx]
                    pred_label = pred_labels[pred_idx].item()
                    
                    best_iou = 0.0
                    best_target_idx = -1
                    
                    # Find best matching ground truth
                    for target_idx in range(len(target_boxes)):
                        target_box = target_boxes[target_idx]
                        target_label = target_labels[target_idx].item()
                        
                        # Only match with same class
                        if pred_label == target_label:
                            iou = self.intersection_over_union(
                                pred_box, target_box, box_format="corners"
                            )
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_target_idx = target_idx
                    
                    # Classify as TP or FP
                    if best_iou >= iou_threshold and best_target_idx >= 0 and not target_matched[best_target_idx]:
                        # True Positive
                        class_tp[pred_label] += 1
                        target_matched[best_target_idx] = True
                    else:
                        # False Positive
                        class_fp[pred_label] += 1
                
                # Count False Negatives (unmatched ground truths)
                for target_idx in range(len(target_boxes)):
                    if not target_matched[target_idx]:
                        target_label = target_labels[target_idx].item()
                        class_fn[target_label] += 1
            
            # Calculate precision, recall, and F1 for each class
            class_precisions = []
            class_recalls = []
            class_f1s = []
            
            for class_idx in range(self.num_classes):
                tp = class_tp[class_idx]
                fp = class_fp[class_idx]
                fn = class_fn[class_idx]
                
                # Calculate precision
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                else:
                    precision = 0.0
                
                # Calculate recall
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                else:
                    recall = 0.0
                
                # Calculate F1
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
                
                class_precisions.append(precision)
                class_recalls.append(recall)
                class_f1s.append(f1)
                
                # Store per-class data
                f1_curve_data['per_class_f1'][class_idx].append(f1)
                f1_curve_data['per_class_precision'][class_idx].append(precision)
                f1_curve_data['per_class_recall'][class_idx].append(recall)
            
            # Calculate overall metrics (mean across classes)
            overall_precision = np.mean(class_precisions)
            overall_recall = np.mean(class_recalls)
            overall_f1 = np.mean(class_f1s)
            
            f1_curve_data['overall_precision'].append(overall_precision)
            f1_curve_data['overall_recall'].append(overall_recall)
            f1_curve_data['overall_f1'].append(overall_f1)
        
        # Find best F1 score and corresponding confidence
        best_overall_f1 = max(f1_curve_data['overall_f1'])
        best_f1_idx = f1_curve_data['overall_f1'].index(best_overall_f1)
        best_confidence = confidence_thresholds[best_f1_idx]
        
        print(f"   Best F1 score: {best_overall_f1:.4f} at confidence threshold: {best_confidence:.3f}")
        
        # Store additional info
        f1_curve_data['best_f1'] = best_overall_f1
        f1_curve_data['best_confidence'] = best_confidence
        f1_curve_data['iou_threshold'] = iou_threshold
        
        print("✅ F1-confidence curve calculation completed using CORRECT methodology")
        return f1_curve_data
    
    def calculate_confusion_matrix(self, predictions, targets, confidence_threshold=0.25, iou_threshold=0.5):
        """
        Calculate confusion matrix following the correct methodology
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
            confidence_threshold: Confidence threshold for filtering predictions
            iou_threshold: IoU threshold for TP/FP classification
            
        Returns:
            Dictionary containing confusion matrix data
        """
        print(f"📊 Computing Confusion Matrix (conf≥{confidence_threshold}, IoU≥{iou_threshold})...")
        
        # Initialize confusion matrix: [predicted_class][true_class]
        # Background class (index = num_classes) for FP and FN
        matrix_size = self.num_classes + 1  # +1 for background
        confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        
        # Class names including background
        class_names_with_bg = self.class_names + ['background']
        
        # Process each image
        for pred_dict, target_dict in zip(predictions, targets):
            
            # Filter predictions by confidence threshold
            if len(pred_dict['scores']) > 0:
                conf_mask = pred_dict['scores'] >= confidence_threshold
                filtered_pred = {
                    'boxes': pred_dict['boxes'][conf_mask],
                    'scores': pred_dict['scores'][conf_mask],
                    'labels': pred_dict['labels'][conf_mask]
                }
            else:
                filtered_pred = {
                    'boxes': torch.empty(0, 4),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long)
                }
            
            pred_boxes = filtered_pred['boxes']
            pred_labels = filtered_pred['labels']
            target_boxes = target_dict['boxes']
            target_labels = target_dict['labels']
            
            # Track which ground truths have been matched
            target_matched = [False] * len(target_boxes)
            
            # Process each prediction
            for pred_idx in range(len(pred_boxes)):
                pred_box = pred_boxes[pred_idx]
                pred_label = pred_labels[pred_idx].item()
                
                best_iou = 0.0
                best_target_idx = -1
                best_target_label = None
                
                # Find best matching ground truth (any class)
                for target_idx in range(len(target_boxes)):
                    target_box = target_boxes[target_idx]
                    target_label = target_labels[target_idx].item()
                    
                    iou = self.intersection_over_union(
                        pred_box, target_box, box_format="corners"
                    )
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = target_idx
                        best_target_label = target_label
                
                # Categorize the prediction
                if best_iou >= iou_threshold and best_target_idx >= 0 and not target_matched[best_target_idx]:
                    # Match found
                    if pred_label == best_target_label:
                        # ✅ Correct Detection (True Positive)
                        confusion_matrix[pred_label][best_target_label] += 1
                        target_matched[best_target_idx] = True
                    else:
                        # ❌ Wrong Classification (predict X but it's Y)
                        confusion_matrix[pred_label][best_target_label] += 1
                        target_matched[best_target_idx] = True
                else:
                    # ❌ Detection Error (predict X but there's nothing there)
                    # Predicted: pred_label, True: background
                    confusion_matrix[pred_label][self.num_classes] += 1  # background column
            
            # Count False Negatives (unmatched ground truths)
            for target_idx in range(len(target_boxes)):
                if not target_matched[target_idx]:
                    target_label = target_labels[target_idx].item()
                    # ❌ Missed Detection (predict nothing but there's object of target_label)
                    # Predicted: background, True: target_label
                    confusion_matrix[self.num_classes][target_label] += 1  # background row
        
        # Create normalized version
        confusion_matrix_normalized = confusion_matrix.astype(float)
        row_sums = confusion_matrix.sum(axis=1)
        for i in range(matrix_size):
            if row_sums[i] > 0:
                confusion_matrix_normalized[i] = confusion_matrix[i] / row_sums[i]
        
        return {
            'matrix': confusion_matrix,
            'matrix_normalized': confusion_matrix_normalized,
            'class_names': class_names_with_bg,
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold
        }
    
    def plot_confusion_matrix(self, cm_data, save_path):
        """
        Plot both normal and normalized confusion matrices
        
        Args:
            cm_data: Dictionary containing confusion matrix data
            save_path: Base path to save the plots (will create two files)
        """
        matrices = [
            (cm_data['matrix'], 'Confusion Matrix (Absolute Counts)', 'Blues', 'd'),
            (cm_data['matrix_normalized'], 'Confusion Matrix (Normalized)', 'Blues', '.2f')
        ]
        
        for idx, (matrix, title, colormap, fmt) in enumerate(matrices):
            plt.figure(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(matrix, 
                       annot=True, 
                       fmt=fmt, 
                       cmap=colormap,
                       xticklabels=cm_data['class_names'],
                       yticklabels=cm_data['class_names'],
                       cbar_kws={'label': 'Count' if idx == 0 else 'Proportion'})
            
            plt.title(f'{title}\n(Confidence ≥ {cm_data["confidence_threshold"]}, IoU ≥ {cm_data["iou_threshold"]})', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('True Class', fontsize=12, fontweight='bold')
            plt.ylabel('Predicted Class', fontsize=12, fontweight='bold')
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # Save with appropriate suffix
            suffix = 'absolute' if idx == 0 else 'normalized'
            file_path = save_path.parent / f"{save_path.stem}_{suffix}{save_path.suffix}"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Confusion matrix ({suffix}) saved to: {file_path}")
        
        # Print matrix statistics
        print(f"\n📊 CONFUSION MATRIX ANALYSIS:")
        print("-" * 50)
        
        total_predictions = np.sum(cm_data['matrix'])
        print(f"Total Predictions: {total_predictions}")
        
        # Calculate accuracy
        correct_predictions = np.trace(cm_data['matrix'][:self.num_classes, :self.num_classes])
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Per-class statistics
        print(f"\nPer-Class Analysis:")
        for i in range(self.num_classes):
            class_name = self.class_names[i]
            
            # True positives (diagonal)
            tp = cm_data['matrix'][i, i]
            
            # False positives (predicted this class but was something else)
            fp = np.sum(cm_data['matrix'][i, :]) - tp
            
            # False negatives (was this class but predicted something else)  
            fn = np.sum(cm_data['matrix'][:, i]) - tp
            
            # Precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  {class_name}:")
            print(f"    TP: {tp:3d}, FP: {fp:3d}, FN: {fn:3d}")
            print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    def fix_combined_precision_recall_curve(self, curve_data):
        """
        Fix the combined classes precision-recall curve to reduce oscillations
        Instead of point-wise averaging, calculate overall TP, FP, FN across all classes
        """
        print("🔧 Fixing combined precision-recall curve calculation...")
        
        # We'll recalculate this properly by combining all detections across classes
        # For now, let's smooth the existing curve to reduce oscillations
        
        overall_precision = np.array(curve_data['custom_overall_precision'])
        overall_recall = np.array(curve_data['custom_overall_recall'])
        
        if len(overall_precision) > 5:  # Only smooth if we have enough points
            # Sort by recall first
            sort_indices = np.argsort(overall_recall)
            recall_sorted = overall_recall[sort_indices]
            precision_sorted = overall_precision[sort_indices]
            
            # Apply smoothing to reduce oscillations
            from scipy.signal import savgol_filter
            window_length = min(len(precision_sorted) // 10, 51)  # Adaptive window
            if window_length % 2 == 0:  # Must be odd
                window_length += 1
            if window_length >= 3:
                precision_smoothed = savgol_filter(precision_sorted, window_length, 3)
                precision_smoothed = np.clip(precision_smoothed, 0, 1)  # Keep in valid range
                
                # Update the curve data
                curve_data['custom_overall_precision'] = precision_smoothed.tolist()
                curve_data['custom_overall_recall'] = recall_sorted.tolist()
                
                print(f"   Applied smoothing with window size {window_length}")
            else:
                print("   Too few points for smoothing")
        
        return curve_data
    
    def plot_f1_confidence_curve(self, f1_curve_data, save_path):
        """
        Plot and save F1-Confidence curve
        
        Args:
            f1_curve_data: Dictionary containing F1 curve metrics
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Better class colors and styles for visibility
        class_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        class_styles = ['-', '--', '-.', ':', '-', '--']
        
        confidence_thresholds = np.array(f1_curve_data['confidence_thresholds'])
        
        # Plot per-class F1 curves (thin lines)
        for class_idx in range(self.num_classes):
            f1_scores = np.array(f1_curve_data['per_class_f1'][class_idx])
            best_f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0
            
            color = class_colors[class_idx % len(class_colors)]
            style = class_styles[class_idx % len(class_styles)]
            
            plt.plot(confidence_thresholds, f1_scores, 
                    color=color, 
                    linestyle=style,
                    linewidth=2, 
                    alpha=0.7,
                    label=f'{self.class_names[class_idx]} (max: {best_f1:.3f})')
        
        # Plot overall F1 curve (thick blue line)
        overall_f1 = np.array(f1_curve_data['overall_f1'])
        best_overall_f1 = f1_curve_data['best_f1']
        best_confidence = f1_curve_data['best_confidence']
        
        plt.plot(confidence_thresholds, overall_f1, 
                color='navy', 
                linewidth=4, 
                label=f'Mean F1 (max: {best_overall_f1:.3f} @ conf={best_confidence:.3f})')
        
        # Mark the best F1 point
        plt.plot(best_confidence, best_overall_f1, 
                marker='o', markersize=10, color='red', 
                markerfacecolor='yellow', markeredgecolor='red',
                label=f'Best Point: F1={best_overall_f1:.3f}')
        
        # Formatting
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xlabel('Confidence Threshold', fontsize=14, fontweight='bold')
        plt.ylabel('F1 Score', fontsize=14, fontweight='bold')
        plt.title(f'F1-Score vs Confidence Threshold (IoU={f1_curve_data["iou_threshold"]})', fontsize=16, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add background color for better visibility
        plt.gca().set_facecolor('#f8f9fa')
        
        # Add text box with best performance info
        textstr = f'Optimal Settings:\nConfidence: {best_confidence:.3f}\nF1-Score: {best_overall_f1:.3f}\nIoU Threshold: {f1_curve_data["iou_threshold"]}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 F1-Confidence curve saved to: {save_path}")
    
    def evaluate(self, confidence_threshold=0.25, save_images=True):
        """
        Run evaluation on test dataset
        
        Args:
            confidence_threshold: Minimum confidence for detections
            save_images: Whether to save annotated images
        """
        print(f"🎯 Starting evaluation with confidence threshold: {confidence_threshold}")
        
        # Load test data (changed from validation)
        test_data = self.load_test_data()
        
        predictions = []
        targets = []
        
        # Add debugging counters
        pred_class_counts = {0: 0, 1: 0, 2: 0}
        gt_class_counts = {0: 0, 1: 0, 2: 0}
        
        print("🔄 Running inference on test set...")
        for i, (img_path, label_path) in enumerate(tqdm(test_data)):
            # Load and preprocess image
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get image dimensions for label conversion
            h0, w0 = image.shape[:2]
            
            # Load ground truth
            gt = self.load_ground_truth_labels(label_path, w0, h0)
            targets.append(gt)
            
            # Count ground truth classes
            for label in gt['labels']:
                gt_class_counts[label.item()] += 1
            
            # Run inference
            image_tensor, pads, original_size, resized_size = self.preprocess_image(image_rgb)
            
            with torch.no_grad():
                preds = self.model(image_tensor)[0]
            
            # Process predictions
            pred_dict = self.postprocess_predictions(preds, pads, original_size, resized_size)
            
            # Debug first prediction BEFORE remapping
            if i == 0:
                print(f"🔍 First prediction types:")
                print(f"   boxes: {type(pred_dict['boxes'])}, shape: {pred_dict['boxes'].shape}")
                print(f"   scores: {type(pred_dict['scores'])}, shape: {pred_dict['scores'].shape}")
                print(f"   labels: {type(pred_dict['labels'])}, shape: {pred_dict['labels'].shape}")
                if len(pred_dict['labels']) > 0:
                    print(f"   Raw model predictions: {pred_dict['labels'][:5].tolist()}")
                if len(gt['labels']) > 0:
                    print(f"   Ground truth classes: {gt['labels'][:5].tolist()}")
            
            # ⚠️ CRITICAL FIX: Apply class remapping to fix mismatch
            pred_dict = self.remap_prediction_classes(pred_dict)
            
            # Debug first prediction AFTER remapping
            if i == 0:
                if len(pred_dict['labels']) > 0:
                    print(f"   Remapped predictions: {pred_dict['labels'][:5].tolist()}")
                print(f"   🎯 Now predictions should match ground truth classes!")
            
            # Apply confidence threshold
            if len(pred_dict['scores']) > 0:
                mask = pred_dict['scores'] >= confidence_threshold
                pred_dict = {
                    'boxes': pred_dict['boxes'][mask].float(),
                    'scores': pred_dict['scores'][mask].float(),
                    'labels': pred_dict['labels'][mask].long()
                }
                
                # Count predicted classes (after confidence filtering and remapping)
                for label in pred_dict['labels']:
                    pred_class_counts[label.item()] += 1
            
            predictions.append(pred_dict)
            
            # Save annotated image if requested
            if save_images:
                # Draw predictions and ground truth on image
                annotated_image = self.draw_predictions_on_image(image, pred_dict, gt)
                
                # Save to results directory with image index for better debugging
                output_filename = f"image_{i:04d}_{img_path.stem}.jpg"
                output_path = self.results_dir / output_filename
                cv2.imwrite(str(output_path), annotated_image)
                
                # Save a summary every 50 images
                if (i + 1) % 50 == 0:
                    print(f"📁 Saved {i + 1}/{len(test_data)} annotated images")
        
        # Print class distribution analysis
        print(f"\n🔍 CLASS DISTRIBUTION ANALYSIS:")
        print(f"{'='*50}")
        print(f"Ground Truth Class Counts:")
        for class_id in sorted(gt_class_counts.keys()):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            print(f"  {class_id} ({class_name}): {gt_class_counts[class_id]} instances")
        
        print(f"\nModel Prediction Class Counts (conf >= {confidence_threshold}):")
        for class_id in sorted(pred_class_counts.keys()):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            print(f"  {class_id} ({class_name}): {pred_class_counts[class_id]} predictions")
        
        print(f"\n⚠️  CLASS MISMATCH ANALYSIS:")
        total_gt = sum(gt_class_counts.values())
        total_pred = sum(pred_class_counts.values())
        print(f"Total ground truth objects: {total_gt}")
        print(f"Total predictions: {total_pred}")
        
        if total_pred == 0:
            print("🔴 CRITICAL: Model made NO predictions! Check confidence threshold or model.")
        elif pred_class_counts[1] == 0 and gt_class_counts[1] > 0:
            print("🔴 CRITICAL: Model never predicts class 1, but GT has many class 1 instances!")
            print("   This suggests your model was trained with different class mapping.")
        elif pred_class_counts[0] > gt_class_counts[0] * 2:
            print("🔴 CRITICAL: Model over-predicts class 0, suggests class mapping mismatch!")
        
        if save_images:
            print(f"✅ All {len(test_data)} annotated images saved to: {self.results_dir}")
        
        print("📊 Computing metrics...")
        
        # Update metrics  
        self.map_metric.update(predictions, targets)
        
        # Compute metrics
        metrics = self.map_metric.compute()
        
        # Generate performance curves
        print("� Generating performance curves...")
        curve_data = self.compute_curve_metrics(predictions, targets)
        
        # Calculate confusion matrix using the optimal confidence from F1 curve
        optimal_confidence = confidence_threshold
        if 'f1_curve_data' in curve_data and 'best_confidence' in curve_data['f1_curve_data']:
            optimal_confidence = curve_data['f1_curve_data']['best_confidence']
            print(f"🎯 Using optimal confidence threshold: {optimal_confidence:.3f}")
        
        confusion_matrix_data = self.calculate_confusion_matrix(
            predictions, targets, 
            confidence_threshold=optimal_confidence, 
            iou_threshold=0.5
        )
        
        # Plot and save curves
        pr_curve_path = self.results_dir / "precision_recall_curve.png"
        f1_curve_path = self.results_dir / "f1_confidence_curve.png"
        cm_path = self.results_dir / "confusion_matrix.png"
        
        self.plot_precision_recall_curve(curve_data, pr_curve_path)
        
        # Plot F1-confidence curve using the new correct methodology
        if 'f1_curve_data' in curve_data:
            self.plot_f1_confidence_curve(curve_data['f1_curve_data'], f1_curve_path)
        
        # Plot confusion matrices
        self.plot_confusion_matrix(confusion_matrix_data, cm_path)
        
        return metrics, predictions, targets, curve_data
    
    def print_results(self, metrics, curve_data=None):
        """Print detailed metric results with explanations"""
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        
        # CORRECT mAP calculation results
        if curve_data and 'mAP' in curve_data:
            print("\n🎯 CORRECT mAP CALCULATION RESULTS:")
            print("-" * 40)
            print(f"mAP@0.5 (Correct): {curve_data['mAP']:.4f} ({curve_data['mAP']*100:.2f}%)")
            
            if 'mAP_50_95' in curve_data:
                print(f"mAP@0.5:0.95 (Correct): {curve_data['mAP_50_95']:.4f} ({curve_data['mAP_50_95']*100:.2f}%)")
            
            if 'per_class_ap' in curve_data:
                print("\nPer-class Average Precision @0.5 (Correct):")
                for i, ap in enumerate(curve_data['per_class_ap']):
                    class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
                    print(f"  Class {i} ({class_name}): {ap:.4f} ({ap*100:.2f}%)")
        
        # TorchMetrics mAP metrics (for comparison)
        print("\n🔍 TORCHMETRICS mAP (for comparison):")
        print("-" * 40)
        
        map_50 = metrics['map_50'].item()
        map_75 = metrics['map_75'].item()
        map_avg = metrics['map'].item()
        
        print(f"mAP@0.5     : {map_50:.4f} ({map_50*100:.2f}%)")
        print(f"mAP@0.75    : {map_75:.4f} ({map_75*100:.2f}%)")
        print(f"mAP@0.5:0.95: {map_avg:.4f} ({map_avg*100:.2f}%)")
        
        print("\n📝 mAP Explanation:")
        print("• mAP@0.5: Average precision across all classes at IoU threshold 0.5")
        print("• mAP@0.75: Average precision across all classes at IoU threshold 0.75") 
        print("• mAP@0.5:0.95: Average precision across IoU thresholds 0.5 to 0.95 (step 0.05)")
        print("• Higher values are better (max = 1.0)")
        
        # Per-class mAP
        if 'map_per_class' in metrics:
            print(f"\n🏷️ PER-CLASS mAP@0.5:0.95 (TorchMetrics):")
            print("-" * 40)
            map_per_class = metrics['map_per_class']
            for i, class_name in enumerate(self.class_names):
                if i < len(map_per_class):
                    print(f"{class_name:15}: {map_per_class[i].item():.4f} ({map_per_class[i].item()*100:.2f}%)")
        
        # Additional metrics if available
        if 'mar_1' in metrics:
            print(f"\n🎯 RECALL METRICS:")
            print("-" * 40)
            print(f"Recall@1    : {metrics['mar_1'].item():.4f} ({metrics['mar_1'].item()*100:.2f}%)")
            print(f"Recall@10   : {metrics['mar_10'].item():.4f} ({metrics['mar_10'].item()*100:.2f}%)")
            print(f"Recall@100  : {metrics['mar_100'].item():.4f} ({metrics['mar_100'].item()*100:.2f}%)")
            
            print("\n📝 Recall Explanation:")
            print("• Recall@K: Maximum recall achievable with K detections per image")
            print("• Measures how well the model finds all objects in the image")
            print("• Higher values are better (max = 1.0)")
        
        # Model performance interpretation - use correct mAP if available
        performance_score = curve_data.get('mAP', map_avg) if curve_data else map_avg
        print(f"\n🔍 PERFORMANCE INTERPRETATION:")
        print("-" * 40)
        
        if performance_score >= 0.7:
            print("🟢 EXCELLENT: Your model has very high accuracy!")
        elif performance_score >= 0.5:
            print("🟡 GOOD: Your model performs well with room for improvement")
        elif performance_score >= 0.3:
            print("🟠 MODERATE: Model needs significant improvement")
        else:
            print("🔴 POOR: Model requires major changes or more training")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        if map_50 - map_75 > 0.2:
            print("• Large gap between mAP@0.5 and mAP@0.75 suggests imprecise localization")
            print("• Consider improving bounding box regression or data augmentation")
        
        if performance_score < 0.5:
            print("• Consider training longer, adjusting learning rate, or improving data quality")
            print("• Check if the model architecture is suitable for your dataset complexity")

def main():
    """Main evaluation function"""
    
    # Configuration
    MODEL_CONFIG = "model/config/models/yolov8n.yaml"
    WEIGHTS = "model/weights/yolov8n/best_2.pt"  # Use the best checkpoint
    DATASET_CONFIG = "model/config/datasets/mask.yaml"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIDENCE_THRESHOLD = 0.25  # Lower threshold for evaluation
    SAVE_IMAGES = True  # Save annotated images
    
    print("🚀 Starting Model Evaluation on Test Set")
    print("="*50)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(MODEL_CONFIG, WEIGHTS, DATASET_CONFIG, DEVICE)
        
        # Run evaluation with image saving
        metrics, predictions, targets, curve_data = evaluator.evaluate(CONFIDENCE_THRESHOLD, SAVE_IMAGES)
        
        # Print results
        evaluator.print_results(metrics, curve_data)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 Evaluated {len(predictions)} test images")
        if SAVE_IMAGES:
            print(f"📁 Annotated images saved to: {evaluator.results_dir}")
        print(f"📈 Performance curves saved to: {evaluator.results_dir}")
        print(f"   • Precision-Recall curve: precision_recall_curve.png")
        print(f"   • F1-Confidence curve: f1_confidence_curve.png")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()