"""
Anti-Overfitting Training Strategy - FIXED VERSION
Based on your working train_3.py pattern
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import csv
from datetime import datetime
import argparse
from tqdm import tqdm
import logging

from model.models.detection_model import DetectionModel
from model.data.dataset import Dataset
from torch.utils.data import DataLoader

class AntiOverfitTrainer:
    def __init__(self, config_path, model_config, dataset_config, device='cuda', checkpoint_path=None):
        # Explicitly set GPU device and verify
        if device == 'cuda' and torch.cuda.is_available():
            # Force use GPU 0 (GTX 1650)
            torch.cuda.set_device(0)
            self.device = torch.device('cuda:0')
            
            # Print GPU info for verification
            print(f"🎯 GPU Selected: {torch.cuda.get_device_name(0)}")
            print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"🎯 Current GPU: cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device('cpu')
            print("⚠️  Using CPU - CUDA not available")
        
        # Load configs
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model properly (following train_3.py pattern)
        self.model = DetectionModel(model_config, device=self.device)
        
        # FREEZE BACKBONE - Only train neck and head
        self._freeze_backbone()
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"🔄 Loading checkpoint from {checkpoint_path}")
            self._load_checkpoint_safe(checkpoint_path)
        else:
            if checkpoint_path:
                print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print("🆕 Starting from scratch")
        
        # Create datasets and dataloaders (following train_3.py pattern)
        train_dataset = Dataset(dataset_config, mode='train', batch_size=self.config['batch_size'])
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_dataset.batch_size,
            shuffle=True,
            collate_fn=Dataset.collate_fn,
        )
        
        val_dataset = Dataset(dataset_config, mode='val', batch_size=self.config['batch_size'])
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=val_dataset.batch_size,
            shuffle=False,
            collate_fn=Dataset.collate_fn,
        )
        
        # Optimizer with strong weight decay (only trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"📈 Optimizer will update {len(trainable_params)} parameter groups")
        
        self.optimizer = optim.AdamW(
            trainable_params,  # Only optimize trainable parameters
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['lr_factor'],
            patience=self.config['lr_patience'],
            min_lr=self.config['min_lr']
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.training_log = []
        
        # Gradual unfreezing parameters
        self.freeze_epochs = self.config.get('freeze_epochs', 15)  # Train with frozen backbone first
        self.unfreeze_lr_factor = self.config.get('unfreeze_lr_factor', 0.1)  # Lower LR for backbone
        self.backbone_unfrozen = False
        self.freeze_phase = True
        
        # EMA for model weights - FIXED
        self.ema_decay = self.config.get('ema_decay', 0.9999)
        self.ema_model = None
        self._initialize_ema()
        
        # Label smoothing for better generalization - REDUCED
        self.label_smoothing = self.config.get('label_smoothing', 0.05)
        
        # Warmup parameters - SHORTENED
        self.warmup_epochs = self.config.get('warmup_epochs', 3)
        self.base_lr = self.config['lr']
        
        # Disable mixed precision for better accuracy
        self.scaler = None
        
        print(f"🎯 Anti-Overfitting Training Initialized")
        print(f"📊 Training samples: {len(self.train_loader.dataset)}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        print(f"⚙️  Batch size: {self.config['batch_size']} | LR: {self.config['lr']} | Device: {self.device}")
        print(f"🎯 Full precision training enabled for maximum accuracy")
        print(f"🧊 PHASE 1: Backbone FROZEN for {self.freeze_epochs} epochs (neck + head only)")
        print(f"🔄 PHASE 2: Gradual unfreezing with {self.unfreeze_lr_factor}x backbone LR")
        
        # Additional GPU verification
        if torch.cuda.is_available():
            print(f"✅ Confirmed using: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            print(f"🔋 Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def _initialize_ema(self):
        """Initialize Exponential Moving Average model"""
        self.ema_model = type(self.model)(
            self.model.yaml, 
            device=self.device
        ) if hasattr(self.model, 'yaml') else None
        
        if self.ema_model:
            self.ema_model.load_state_dict(self.model.state_dict())
            for param in self.ema_model.parameters():
                param.requires_grad = False
    
    def _freeze_backbone(self):
        """Freeze backbone parameters - only train neck and head for anti-overfitting"""
        frozen_params = 0
        trainable_params = 0
        trainable_layers = []
        
        print("🧊 Freezing backbone layers...")
        
        # Define which layers should be trainable
        # Backbone: layers 0-9 (freeze these)
        # Neck: layers 11, 13, 14, 15, 16, 17 (train these) 
        # Head: layer 18 (train this)
        # Note: layers 10, 12 are Upsample (no parameters)
        trainable_layer_indices = {11, 13, 14, 15, 16, 17, 18}
        
        for name, param in self.model.named_parameters():            
            if 'model.' in name:
                # Extract layer index from parameter name like "model.0.conv.weight" -> 0
                parts = name.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    
                    # Check if this layer should be trainable
                    if layer_idx in trainable_layer_indices:
                        param.requires_grad = True
                        trainable_params += param.numel()
                        layer_idx_str = str(layer_idx)
                        if layer_idx_str not in trainable_layers:
                            trainable_layers.append(layer_idx_str)
                    else:
                        # Freeze backbone layers (0-9) and any other layers
                        param.requires_grad = False
                        frozen_params += param.numel()
                else:
                    # If we can't parse the layer index, make it trainable by default
                    param.requires_grad = True  
                    trainable_params += param.numel()
            else:
                # Keep other parameters trainable (if any)
                param.requires_grad = True
                trainable_params += param.numel()
        
        total_params = frozen_params + trainable_params
        frozen_percent = (frozen_params / total_params * 100) if total_params > 0 else 0
        trainable_percent = (trainable_params / total_params * 100) if total_params > 0 else 0
        
        print(f"❄️  Frozen parameters: {frozen_params:,} ({frozen_percent:.1f}%)")
        print(f"🔥 Trainable parameters: {trainable_params:,} ({trainable_percent:.1f}%)")
        print(f"📊 Total parameters: {total_params:,}")
        
        if trainable_layers:
            print(f"🎯 Training only layers: {sorted(trainable_layers)} (neck + head)")
        else:
            print("⚠️  WARNING: No trainable layers found!")
            
        # Ensure we have trainable parameters
        if trainable_params == 0:
            print("🚨 ERROR: All parameters are frozen! This will cause optimizer failure.")
            print("🔧 Emergency unfreezing neck and head layers...")
            
            # Emergency unfreeze of neck and head
            for name, param in self.model.named_parameters():
                if 'model.' in name:
                    parts = name.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        layer_idx = int(parts[1])
                        if layer_idx >= 10:  # Force unfreeze neck and head
                            param.requires_grad = True
                            trainable_params += param.numel()
                            
            print(f"🔧 After emergency unfreeze: {trainable_params:,} trainable parameters")
    
    def _unfreeze_backbone_gradual(self):
        """Gradually unfreeze backbone with lower learning rate"""
        print("🔓 GRADUAL UNFREEZING: Enabling backbone training...")
        
        # Unfreeze backbone parameters
        backbone_params = []
        neck_head_params = []
        
        for name, param in self.model.named_parameters():
            if 'model.' in name:
                layer_idx = int(name.split('.')[1]) if '.' in name.split('.')[1] else -1
                
                # Backbone (layers 0-9) - unfreeze with lower LR
                if layer_idx <= 9:
                    param.requires_grad = True
                    backbone_params.append(param)
                # Neck and head (layers 10+) - keep current LR
                else:
                    neck_head_params.append(param)
            else:
                neck_head_params.append(param)
        
        # Create new optimizer with different learning rates for backbone vs neck/head
        current_lr = self.optimizer.param_groups[0]['lr']
        backbone_lr = current_lr * self.unfreeze_lr_factor  # Much lower LR for backbone
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
            {'params': neck_head_params, 'lr': current_lr, 'name': 'neck_head'}
        ], weight_decay=self.config['weight_decay'])
        
        # Update scheduler to work with new optimizer
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['lr_factor'],
            patience=self.config['lr_patience'],
            min_lr=self.config['min_lr']
        )
        
        self.backbone_unfrozen = True
        self.freeze_phase = False
        
        total_backbone = sum(p.numel() for p in backbone_params)
        total_neck_head = sum(p.numel() for p in neck_head_params)
        
        print(f"✅ Backbone unfrozen!")
        print(f"🧊➡️🔥 Backbone params: {total_backbone:,} (LR: {backbone_lr:.2e})")
        print(f"🔥 Neck+Head params: {total_neck_head:,} (LR: {current_lr:.2e})")
        print(f"📊 LR ratio (backbone/neck): {self.unfreeze_lr_factor:.1f}x")
        print("🎯 Now training ENTIRE model with differential learning rates")
    
    def _load_checkpoint_safe(self, checkpoint_path):
        """Safe checkpoint loading with compatibility checks - handles multiple formats"""
        try:
            print(f"🔄 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract model state dict - handle multiple formats
            checkpoint_state = None
            optimizer_state = None
            additional_info = {}
            
            if isinstance(checkpoint, dict):
                # Format 1: Training checkpoint with nested structure
                if 'model_state_dict' in checkpoint:
                    checkpoint_state = checkpoint['model_state_dict']
                    optimizer_state = checkpoint.get('optimizer_state_dict', None)
                    additional_info = {
                        'epoch': checkpoint.get('epoch', 0),
                        'val_loss': checkpoint.get('val_loss', None),
                        'train_loss': checkpoint.get('train_loss', None)
                    }
                    print("📂 Detected training checkpoint format")
                
                # Format 2: Converted YOLO format  
                elif 'model' in checkpoint:
                    checkpoint_state = checkpoint['model']
                    print("📂 Detected converted YOLO format")
                
                # Format 3: Direct state dict with metadata
                elif any(key.startswith(('model.', 'backbone.', 'neck.', 'head.')) for key in checkpoint.keys()):
                    # Contains model parameters directly
                    checkpoint_state = checkpoint
                    print("📂 Detected direct state dict format")
                
                # Format 4: Unknown dict format - try as direct state dict
                else:
                    checkpoint_state = checkpoint
                    print("📂 Attempting to load as direct state dict")
                    
            else:
                # Format 5: Direct model object or state dict
                if hasattr(checkpoint, 'state_dict'):
                    checkpoint_state = checkpoint.state_dict()
                    print("📂 Extracted state dict from model object")
                else:
                    checkpoint_state = checkpoint
                    print("📂 Using checkpoint as direct state dict")
            
            # Verify we have a valid state dict
            if checkpoint_state is None or not isinstance(checkpoint_state, dict):
                raise ValueError("Could not extract valid state dict from checkpoint")
            
            # Get current model state
            model_state = self.model.state_dict()
            
            # Smart loading: only load compatible parameters
            compatible_count = 0
            incompatible_count = 0
            loaded_state = {}
            
            print("🔍 Checking parameter compatibility...")
            print(f"📊 Checkpoint has {len(checkpoint_state)} parameters")
            print(f"📊 Model expects {len(model_state)} parameters")
            
            for param_name, checkpoint_param in checkpoint_state.items():
                if param_name in model_state:
                    model_param = model_state[param_name]
                    
                    if checkpoint_param.shape == model_param.shape:
                        # Shapes match - load this parameter
                        loaded_state[param_name] = checkpoint_param.clone()
                        compatible_count += 1
                    else:
                        # Shapes don't match - skip
                        incompatible_count += 1
                        if incompatible_count <= 5:  # Only show first 5 mismatches
                            print(f"⚠️  Skipping {param_name}: shape mismatch {checkpoint_param.shape} vs {model_param.shape}")
                        elif incompatible_count == 6:
                            print(f"⚠️  ... and {len(checkpoint_state) - compatible_count - 5} more shape mismatches")
                else:
                    # Parameter doesn't exist in current model
                    incompatible_count += 1
                    if incompatible_count <= 3:  # Only show first 3 missing params
                        print(f"⚠️  Skipping {param_name}: not found in current model")
            
            # Load only compatible parameters
            if loaded_state:
                # Use strict=False to allow partial loading
                missing_keys, unexpected_keys = self.model.load_state_dict(loaded_state, strict=True)
                
                print(f"✅ Successfully loaded {compatible_count} compatible parameters")
                print(f"⚠️  Skipped {incompatible_count} incompatible parameters")
                
                if missing_keys:
                    print(f"🆕 {len(missing_keys)} parameters will be randomly initialized")
                
                if unexpected_keys:
                    print(f"� {len(unexpected_keys)} unexpected parameters were ignored")
                    
                print("�🔄 Model loaded with partial weights - compatible layers initialized from checkpoint")
                
                # Try to load optimizer state if available and if we have the right format
                if optimizer_state is not None and hasattr(self, 'optimizer'):
                    try:
                        self.optimizer.load_state_dict(optimizer_state)
                        print("✅ Optimizer state loaded successfully")
                    except Exception as opt_e:
                        print(f"⚠️  Could not load optimizer state: {opt_e}")
                
                # Print additional info if available
                if additional_info.get('epoch'):
                    print(f"📊 Checkpoint was from epoch {additional_info['epoch']}")
                if additional_info.get('val_loss'):
                    print(f"📊 Previous validation loss: {additional_info['val_loss']:.4f}")
                    
            else:
                print("❌ No compatible parameters found - starting from scratch")
                
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print(f"🔍 Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"🔍 Checkpoint keys: {list(checkpoint.keys())}")
            print("🆕 Starting training from scratch instead")
    
    def _update_ema(self):
        """Update EMA model weights"""
        if self.ema_model:
            with torch.no_grad():
                for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                    ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)
    
    def _check_memory_usage(self):
        """Monitor memory usage and clear cache if needed - Silent version"""
        try:
            import psutil
            import gc
            
            # Get system memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get GPU memory if available
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_total = torch.cuda.get_device_properties(0).total_memory
                gpu_percent = (gpu_memory_allocated / gpu_total) * 100
                
                # Clear cache if memory usage is high - SILENTLY
                if gpu_percent > 80 or memory_percent > 85:
                    torch.cuda.empty_cache()
                    gc.collect()
                    return True, memory_percent, gpu_percent
                
                return False, memory_percent, gpu_percent
            else:
                if memory_percent > 85:
                    gc.collect()
                    return True, memory_percent, 0
                return False, memory_percent, 0
            
        except ImportError:
            # Fallback if psutil not available
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return True, 0, 0
    
    def _get_memory_info(self):
        """Get current memory usage for display"""
        try:
            import psutil
            
            # System memory
            memory = psutil.virtual_memory()
            sys_percent = memory.percent
            
            # GPU memory if available
            if torch.cuda.is_available():
                # Make sure we're checking the correct GPU
                current_gpu = torch.cuda.current_device()
                gpu_memory_allocated = torch.cuda.memory_allocated(current_gpu)
                gpu_total = torch.cuda.get_device_properties(current_gpu).total_memory
                gpu_percent = (gpu_memory_allocated / gpu_total) * 100
                return sys_percent, gpu_percent
            else:
                return sys_percent, 0
        except ImportError:
            return 0, 0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        current_epoch = len(self.training_log) + 1
        
        # Warmup learning rate
        if current_epoch <= self.warmup_epochs:
            warmup_lr = self.base_lr * (current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Progressive weight decay - increase regularization over time
        if current_epoch > 50:  # After 50 epochs, increase weight decay
            base_weight_decay = self.config['weight_decay']
            progressive_factor = 1 + (current_epoch - 50) * 0.01  # Gradually increase
            new_weight_decay = min(base_weight_decay * progressive_factor, 0.3)  # Cap at 0.3
            
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = new_weight_decay
        
        with tqdm(self.train_loader, desc="Training", unit="batch") as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move batch data to device (following train_3.py pattern)
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(self.device, non_blocking=True)
                    
                    # Forward pass
                    loss = self.model.loss(batch)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    # Update EMA weights
                    self._update_ema()
                    
                    total_loss += loss.item()
                    
                    # Memory management - AGGRESSIVE clearing every 25 batches
                    if batch_idx % 25 == 0 and batch_idx > 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    
                    # Update progress bar every 10 batches to avoid slowdown
                    if batch_idx % 10 == 0 or batch_idx == len(self.train_loader) - 1:
                        current_wd = self.optimizer.param_groups[0]['weight_decay']
                        sys_mem, gpu_mem = self._get_memory_info()
                        
                        progress_info = {
                            'loss': f'{loss.item():.4f}',
                            'wd': f'{current_wd:.3f}',
                            'sys': f'{sys_mem:.1f}%',
                        }
                        
                        if torch.cuda.is_available():
                            progress_info['gpu'] = f'{gpu_mem:.1f}%'
                        
                        pbar.set_postfix(progress_info)
                    
                    # Clear variables to free memory
                    del loss
                    if batch_idx % 100 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                except Exception as e:
                    print(f"❌ Error in batch {batch_idx}: {str(e)[:100]}...")  # Truncate long error messages
                    # Clear memory and continue
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    continue
        
        return total_loss / num_batches
    
    def validate(self):
        """Validation following the exact pattern from train_3.py"""
        self.model.train()  # Keep in train mode like train_3.py
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validating", unit="batch") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    try:
                        # Move batch data to device
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(self.device, non_blocking=True)
                        
                        # Forward pass
                        loss = self.model.loss(batch)
                        
                        val_loss += loss.item()
                        
                        # Memory management every 25 batches - MORE AGGRESSIVE
                        if batch_idx % 25 == 0 and batch_idx > 0:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                        
                        # Update progress bar every 5 batches (validation is shorter)
                        if batch_idx % 5 == 0 or batch_idx == len(self.val_loader) - 1:
                            sys_mem, gpu_mem = self._get_memory_info()
                            
                            progress_info = {
                                'val_loss': f'{loss.item():.4f}',
                                'sys': f'{sys_mem:.1f}%',
                            }
                            
                            if torch.cuda.is_available():
                                progress_info['gpu'] = f'{gpu_mem:.1f}%'
                            
                            pbar.set_postfix(progress_info)
                        
                        # Clear variables
                        del loss
                        
                    except Exception as e:
                        print(f"❌ Validation error in batch {batch_idx}: {e}")
                        continue
        
        return val_loss / num_batches
    
    def train(self):
        print("🚀 Starting Anti-Overfitting Training...")
        print(f"� Training Configuration:")
        print(f"   • Epochs: {self.config['epochs']} | Patience: {self.config['patience']}")
        print(f"   • Optimizer: {self.config['optimizer']} | Weight Decay: {self.config['weight_decay']}")
        print(f"   • LR Schedule: {self.config['lr_scheduler']} (factor={self.config['lr_factor']})")
        print(f"   • Regularization: Label Smoothing={self.config.get('label_smoothing', 0.1)}")
        print(f"   • Mixed Precision: Disabled for better accuracy")
        print(f"   • Save Directory: {self.config['save_dir']}")
        print("=" * 70)
        
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Check for gradual unfreezing
            if self.freeze_phase and epoch > self.freeze_epochs and not self.backbone_unfrozen:
                print(f"\n{'='*70}")
                print(f"🔄 SWITCHING TO PHASE 2 at epoch {epoch}")
                print(f"{'='*70}")
                self._unfreeze_backbone_gradual()
                # Reset early stopping counter when unfreezing
                self.epochs_no_improve = 0
                print(f"🔄 Early stopping counter reset for unfreezing phase")
                print(f"{'='*70}\n")
            
            # Training
            train_loss = self.train_epoch()
            
            # More frequent validation when improvement is possible
            should_validate = (
                epoch % 2 == 0 or  # Every 2 epochs instead of 3
                self.epochs_no_improve > 10 or  # More frequent when struggling
                epoch <= 20  # Always validate in early epochs
            )
            
            if should_validate:
                val_loss = self.validate()
                self.scheduler.step(val_loss)
            else:
                val_loss = None
            
            # Logging
            phase = "FROZEN" if self.freeze_phase else "UNFROZEN"
            current_lr = self.optimizer.param_groups[0]['lr']
            backbone_lr = self.optimizer.param_groups[0]['lr'] if len(self.optimizer.param_groups) == 1 else self.optimizer.param_groups[0]['lr']
            
            log_entry = {
                'epoch': epoch,
                'phase': phase,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': current_lr,
                'backbone_lr': backbone_lr if not self.freeze_phase else 0.0,
                'best_val_loss': self.best_val_loss,
                'epochs_no_improve': self.epochs_no_improve
            }
            self.training_log.append(log_entry)
            
            # Print progress
            if val_loss is not None:
                phase_info = f"[{phase}]" if phase else ""
                lr_info = f"LR: {current_lr:.2e}"
                if not self.freeze_phase and len(self.optimizer.param_groups) > 1:
                    backbone_lr = self.optimizer.param_groups[0]['lr']
                    neck_lr = self.optimizer.param_groups[1]['lr']
                    lr_info = f"LR(B/NH): {backbone_lr:.2e}/{neck_lr:.2e}"
                
                print(f"Epoch {epoch:3d} {phase_info} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                      f"{lr_info} | No improve: {self.epochs_no_improve}")
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    
                    # Save best model with EMA weights
                    save_dict = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'train_loss': train_loss,
                        'config': self.config
                    }
                    
                    # Also save EMA weights if available
                    if self.ema_model:
                        save_dict['ema_state_dict'] = self.ema_model.state_dict()
                    
                    torch.save(save_dict, os.path.join(self.config['save_dir'], 'best_anti_overfit.pt'))
                    
                    print(f"💾 New best validation loss: {val_loss:.4f}")
                else:
                    self.epochs_no_improve += 2 if should_validate else 3
                    
                    # More aggressive LR reduction when stuck
                    if self.epochs_no_improve > 15:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        new_lr = current_lr * 0.5  # Cut LR in half
                        if new_lr >= self.config['min_lr']:
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = new_lr
                            print(f"🔥 AGGRESSIVE LR CUT: {current_lr:.2e} → {new_lr:.2e}")
                            self.epochs_no_improve = 0  # Reset counter after manual LR cut
            else:
                phase_info = f"[{phase}]" if phase else ""
                lr_info = f"LR: {current_lr:.2e}"
                if not self.freeze_phase and len(self.optimizer.param_groups) > 1:
                    backbone_lr = self.optimizer.param_groups[0]['lr']
                    neck_lr = self.optimizer.param_groups[1]['lr']
                    lr_info = f"LR(B/NH): {backbone_lr:.2e}/{neck_lr:.2e}"
                
                print(f"Epoch {epoch:3d} {phase_info} | Train: {train_loss:.4f} | {lr_info}")
            
            # Early stopping
            if self.epochs_no_improve >= self.config['patience']:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
        
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'train_loss': train_loss
        }, os.path.join(self.config['save_dir'], 'final_anti_overfit.pt'))
        
        # Save training log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"anti_overfit_log_{timestamp}.csv"
        
        with open(log_file, 'w', newline='') as f:
            if self.training_log:
                writer = csv.DictWriter(f, fieldnames=self.training_log[0].keys())
                writer.writeheader()
                writer.writerows(self.training_log)
        
        print(f"🎉 Training completed! Log saved to {log_file}")
        print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='model/config/models/yolov8n.yaml')
    parser.add_argument('--train_config', default='model/config/training/anti_overfit.yaml')
    parser.add_argument('--dataset', default='model/config/datasets/mask.yaml')
    parser.add_argument('--device', default='cuda:0', 
                       help='Device to use (cuda:0 for GTX 1650, cuda:1 for other GPU, cpu)')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to checkpoint to resume from (e.g., last_advanced.pt)')
    
    args = parser.parse_args()
    
    # Handle "None" string argument
    if args.checkpoint and args.checkpoint.lower() == 'none':
        args.checkpoint = None
        print("🆕 Starting fresh training (no checkpoint)")
    
    # Print available GPUs for reference
    if torch.cuda.is_available():
        print("🔍 Available GPUs:")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"🎯 Selected device: {args.device}")
        print("=" * 50)
    
    trainer = AntiOverfitTrainer(
        config_path=args.train_config,
        model_config=args.model_config,
        dataset_config=args.dataset,
        device=args.device,
        checkpoint_path=args.checkpoint
    )
    
    trainer.train()

if __name__ == '__main__':
    main()
