import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

# Import our custom modules
import sys
sys.path.append('..')
from models import MobileDepthNet
from loss.depth_losses import MobileDepthLoss, DepthMetrics
from data.kitti_dataset import create_kitti_dataloaders
from speed_estimation.speed_estimator import SpeedEstimator, SpeedEstimationConfig


class MobileDepthTrainer:
    """
    Comprehensive trainer for MobileDepth model with depth estimation and speed estimation
    """
    
    def __init__(
        self,
        model: MobileDepthNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: MobileDepthLoss,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        log_dir: str = "logs",
        save_dir: str = "checkpoints",
        speed_estimation: bool = False
    ):
        """
        Args:
            model: MobileDepth model
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Training device
            log_dir: Directory for tensorboard logs
            save_dir: Directory for model checkpoints
            speed_estimation: Whether to include speed estimation training
        """
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.speed_estimation = speed_estimation
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup logging
        self.log_dir = log_dir
        self.save_dir = save_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Tensorboard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(self.log_dir, f"mobiledepth_{timestamp}"))
        
        # Speed estimator if needed
        if self.speed_estimation:
            self.speed_estimator = SpeedEstimator().to(self.device)
            self.speed_loss_fn = nn.MSELoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        
        # Metrics tracking
        self.metrics_tracker = DepthMetrics()
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        additional_info: Optional[Dict] = None
    ):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.speed_estimation:
            checkpoint['speed_estimator_state_dict'] = self.speed_estimator.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return epoch"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.speed_estimation and 'speed_estimator_state_dict' in checkpoint:
            self.speed_estimator.load_state_dict(checkpoint['speed_estimator_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.metrics_history = checkpoint.get('metrics_history', [])
        
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")
        
        return epoch
    
    def train_epoch(self) -> Tuple[float, Dict]:
        """Train for one epoch"""
        
        self.model.train()
        if self.speed_estimation:
            self.speed_estimator.train()
        
        epoch_losses = []
        epoch_metrics = []
        loss_components = {
            'total_loss': [],
            'l1_loss': [],
            'ssim_loss': [],
            'gradient_loss': [],
            'scale_inv_loss': []
        }
        
        if self.speed_estimation:
            epoch_speed_losses = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            
            if self.speed_estimation:
                # Speed estimation mode - expect sequence data
                images, depths, time_intervals = batch_data
                # Use first two frames for training
                image = images[0].to(self.device)
                depth = depths[0].to(self.device)
            else:
                # Regular depth estimation
                image, depth = batch_data
                image = image.to(self.device)
                depth = depth.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_depth = self.model(image)
            
            # Compute depth loss
            depth_loss, loss_dict = self.loss_fn(pred_depth, depth)
            total_loss = depth_loss
            
            # Speed estimation loss if enabled
            if self.speed_estimation:
                speed_loss = self._compute_speed_loss(images, depths, pred_depth)
                total_loss += speed_loss
                epoch_speed_losses.append(speed_loss.item())
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Record losses
            epoch_losses.append(total_loss.item())
            for key, value in loss_dict.items():
                loss_components[key].append(value)
            
            # Compute metrics
            with torch.no_grad():
                metrics = self.metrics_tracker.compute_metrics(pred_depth, depth)
                if metrics:
                    epoch_metrics.append(metrics)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'L1': f"{loss_dict['l1_loss']:.4f}",
                'SSIM': f"{loss_dict['ssim_loss']:.4f}"
            })
            
            # Log batch metrics to tensorboard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            if batch_idx % 100 == 0:  # Log every 100 batches
                self.writer.add_scalar('Train/BatchLoss', total_loss.item(), global_step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Train/{key}', value, global_step)
        
        # Aggregate epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_loss_components = {key: np.mean(values) for key, values in loss_components.items()}
        
        if epoch_metrics:
            avg_metrics = {}
            for key in epoch_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        else:
            avg_metrics = {}
        
        epoch_stats = {
            'avg_loss': avg_loss,
            'loss_components': avg_loss_components,
            'metrics': avg_metrics
        }
        
        if self.speed_estimation:
            epoch_stats['speed_loss'] = np.mean(epoch_speed_losses)
        
        return avg_loss, epoch_stats
    
    def validate_epoch(self) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        
        self.model.eval()
        if self.speed_estimation:
            self.speed_estimator.eval()
        
        epoch_losses = []
        epoch_metrics = []
        loss_components = {
            'total_loss': [],
            'l1_loss': [],
            'ssim_loss': [],
            'gradient_loss': [],
            'scale_inv_loss': []
        }
        
        if self.speed_estimation:
            epoch_speed_losses = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch_data in progress_bar:
                
                if self.speed_estimation:
                    images, depths, time_intervals = batch_data
                    image = images[0].to(self.device)
                    depth = depths[0].to(self.device)
                else:
                    image, depth = batch_data
                    image = image.to(self.device)
                    depth = depth.to(self.device)
                
                # Forward pass
                pred_depth = self.model(image)
                
                # Compute depth loss
                depth_loss, loss_dict = self.loss_fn(pred_depth, depth)
                total_loss = depth_loss
                
                # Speed estimation loss if enabled
                if self.speed_estimation:
                    speed_loss = self._compute_speed_loss(images, depths, pred_depth)
                    total_loss += speed_loss
                    epoch_speed_losses.append(speed_loss.item())
                
                # Record losses
                epoch_losses.append(total_loss.item())
                for key, value in loss_dict.items():
                    loss_components[key].append(value)
                
                # Compute metrics
                metrics = self.metrics_tracker.compute_metrics(pred_depth, depth)
                if metrics:
                    epoch_metrics.append(metrics)
                
                progress_bar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'L1': f"{loss_dict['l1_loss']:.4f}",
                    'SSIM': f"{loss_dict['ssim_loss']:.4f}"
                })
        
        # Aggregate epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_loss_components = {key: np.mean(values) for key, values in loss_components.items()}
        
        if epoch_metrics:
            avg_metrics = {}
            for key in epoch_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        else:
            avg_metrics = {}
        
        epoch_stats = {
            'avg_loss': avg_loss,
            'loss_components': avg_loss_components,
            'metrics': avg_metrics
        }
        
        if self.speed_estimation:
            epoch_stats['speed_loss'] = np.mean(epoch_speed_losses)
        
        return avg_loss, epoch_stats
    
    def _compute_speed_loss(self, images: List[torch.Tensor], depths: List[torch.Tensor], pred_depth: torch.Tensor) -> torch.Tensor:
        """Compute speed estimation loss"""
        
        if len(images) < 2 or len(depths) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Use predicted depth for first frame and ground truth for second
        depth_sequence = [pred_depth, depths[1].to(self.device)]
        image_sequence = [img.to(self.device) for img in images[:2]]
        
        # Estimate speed
        speed_estimates, _ = self.speed_estimator(depth_sequence, image_sequence)
        
        # Dummy ground truth speed (in real scenario, you'd have actual speed labels)
        # For now, we'll use a consistency loss that encourages reasonable speed estimates
        target_speeds = torch.ones_like(speed_estimates) * 5.0  # Assume 5 m/s average
        
        speed_loss = self.speed_loss_fn(speed_estimates, target_speeds)
        return speed_loss * 0.1  # Weight the speed loss
    
    def train(
        self,
        num_epochs: int,
        save_every: int = 5,
        validate_every: int = 1,
        early_stopping_patience: int = 10
    ):
        """Main training loop"""
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_loss, train_stats = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            if epoch % validate_every == 0:
                val_loss, val_stats = self.validate_epoch()
                self.val_losses.append(val_loss)
                self.metrics_history.append(val_stats['metrics'])
                
                # Check for best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Log to tensorboard
                self._log_epoch_metrics(epoch, train_stats, val_stats)
                
                # Print epoch summary
                epoch_time = time.time() - start_time
                self._print_epoch_summary(epoch, train_stats, val_stats, epoch_time)
                
                # Save checkpoint
                if epoch % save_every == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
            else:
                # Only training, no validation
                self._log_epoch_metrics(epoch, train_stats, None)
                print(f"Epoch {epoch}/{num_epochs-1} - Train Loss: {train_loss:.4f}")
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        print("Training completed!")
        self.writer.close()
        
        # Save final model
        self.save_checkpoint(self.current_epoch, additional_info={'final_model': True})
    
    def _log_epoch_metrics(self, epoch: int, train_stats: Dict, val_stats: Optional[Dict]):
        """Log metrics to tensorboard"""
        
        # Training metrics
        self.writer.add_scalar('Loss/Train', train_stats['avg_loss'], epoch)
        for key, value in train_stats['loss_components'].items():
            self.writer.add_scalar(f'Loss/Train_{key}', value, epoch)
        
        for key, value in train_stats['metrics'].items():
            self.writer.add_scalar(f'Metrics/Train_{key}', value, epoch)
        
        if 'speed_loss' in train_stats:
            self.writer.add_scalar('Loss/Train_speed', train_stats['speed_loss'], epoch)
        
        # Validation metrics
        if val_stats:
            self.writer.add_scalar('Loss/Val', val_stats['avg_loss'], epoch)
            for key, value in val_stats['loss_components'].items():
                self.writer.add_scalar(f'Loss/Val_{key}', value, epoch)
            
            for key, value in val_stats['metrics'].items():
                self.writer.add_scalar(f'Metrics/Val_{key}', value, epoch)
            
            if 'speed_loss' in val_stats:
                self.writer.add_scalar('Loss/Val_speed', val_stats['speed_loss'], epoch)
        
        # Learning rate
        if self.scheduler:
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
    
    def _print_epoch_summary(self, epoch: int, train_stats: Dict, val_stats: Dict, epoch_time: float):
        """Print epoch summary"""
        
        print(f"\nEpoch {epoch}/{self.current_epoch}")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_stats['avg_loss']:.4f}")
        print(f"Val Loss: {val_stats['avg_loss']:.4f}")
        
        # Key metrics
        if train_stats['metrics']:
            print(f"Train δ1: {train_stats['metrics'].get('delta1', 0):.3f}")
        if val_stats['metrics']:
            print(f"Val δ1: {val_stats['metrics'].get('delta1', 0):.3f}")
            print(f"Val RMSE: {val_stats['metrics'].get('rmse', 0):.3f}")
        
        if 'speed_loss' in train_stats:
            print(f"Speed Loss: {train_stats['speed_loss']:.4f}")
        
        print("-" * 50)
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Metrics over time
        if self.metrics_history:
            epochs = range(len(self.metrics_history))
            
            # Delta1 accuracy
            delta1_values = [m.get('delta1', 0) for m in self.metrics_history]
            axes[0, 1].plot(epochs, delta1_values)
            axes[0, 1].set_title('δ1 Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('δ1')
            axes[0, 1].grid(True)
            
            # RMSE
            rmse_values = [m.get('rmse', 0) for m in self.metrics_history]
            axes[1, 0].plot(epochs, rmse_values)
            axes[1, 0].set_title('RMSE')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('RMSE')
            axes[1, 0].grid(True)
            
            # Absolute relative error
            abs_rel_values = [m.get('abs_rel', 0) for m in self.metrics_history]
            axes[1, 1].plot(epochs, abs_rel_values)
            axes[1, 1].set_title('Absolute Relative Error')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Abs Rel')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_trainer(
    data_dir: str,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    image_size: Tuple[int, int] = (224, 224),
    speed_estimation: bool = False,
    device: Optional[torch.device] = None
) -> MobileDepthTrainer:
    """
    Create a MobileDepth trainer with default configuration
    
    Args:
        data_dir: Path to KITTI dataset
        batch_size: Training batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        image_size: Input image size
        speed_estimation: Whether to include speed estimation
        device: Training device
    
    Returns:
        Configured trainer
    """
    
    # Device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = MobileDepthNet()
    
    # Data loaders
    train_loader, val_loader = create_kitti_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        speed_estimation=speed_estimation
    )
    
    # Loss function
    loss_fn = MobileDepthLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,  # Adjust based on number of epochs
        eta_min=1e-6
    )
    
    # Create trainer
    trainer = MobileDepthTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        speed_estimation=speed_estimation
    )
    
    return trainer


if __name__ == "__main__":
    # Example usage
    data_dir = "/kaggle/input/kitti-depth-prediction-evaluation"
    
    # Create trainer
    trainer = create_trainer(
        data_dir=data_dir,
        batch_size=16,
        learning_rate=1e-4,
        speed_estimation=False
    )
    
    # Train model
    trainer.train(
        num_epochs=50,
        save_every=5,
        validate_every=1,
        early_stopping_patience=10
    )
    
    # Plot training history
    trainer.plot_training_history("training_history.png")