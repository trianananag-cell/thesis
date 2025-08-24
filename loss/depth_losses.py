import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss for depth estimation
    """
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        channel: int = 1,
        reduction: str = 'mean'
    ):
        """
        Args:
            window_size: Size of the sliding window
            sigma: Standard deviation for Gaussian kernel
            data_range: Range of the data (max - min)
            channel: Number of channels
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.channel = channel
        self.reduction = reduction
        
        # Create Gaussian kernel
        self.register_buffer('window', self._create_gaussian_window())
    
    def _create_gaussian_window(self) -> torch.Tensor:
        """Create 2D Gaussian kernel"""
        coords = torch.arange(self.window_size, dtype=torch.float32)
        coords -= self.window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g /= g.sum()
        
        # Create 2D kernel
        kernel = g[:, None] * g[None, :]
        kernel = kernel.expand(self.channel, 1, self.window_size, self.window_size)
        
        return kernel
    
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        channel: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute SSIM between two images"""
        
        # Constants for stability
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        
        # Compute means
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        contrast_sensitivity = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        
        return ssim_map, contrast_sensitivity
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted depth map [B, C, H, W]
            target: Ground truth depth map [B, C, H, W]
        """
        if pred.size() != target.size():
            raise ValueError(f"Input sizes do not match: {pred.size()} vs {target.size()}")
        
        # Ensure window is on the correct device
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)
        
        # Compute SSIM
        ssim_map, _ = self._ssim(pred, target, self.window, self.window_size, self.channel)
        
        # Convert to loss (1 - SSIM)
        ssim_loss = 1 - ssim_map
        
        if self.reduction == 'mean':
            return ssim_loss.mean()
        elif self.reduction == 'sum':
            return ssim_loss.sum()
        else:
            return ssim_loss


class GradientLoss(nn.Module):
    """
    Gradient loss for preserving edge information in depth maps
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted depth map [B, C, H, W]
            target: Ground truth depth map [B, C, H, W]
        """
        # Compute gradients
        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        # L1 loss on gradients
        loss_x = F.l1_loss(pred_grad_x, target_grad_x, reduction='none')
        loss_y = F.l1_loss(pred_grad_y, target_grad_y, reduction='none')
        
        loss = loss_x.mean() + loss_y.mean()
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * pred.numel()
        else:
            return loss


class ScaleInvariantLogLoss(nn.Module):
    """
    Scale-invariant logarithmic loss for depth estimation
    From "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
    """
    
    def __init__(self, alpha: float = 0.5, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for variance term
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted depth map [B, C, H, W]
            target: Ground truth depth map [B, C, H, W]
            mask: Valid pixel mask [B, C, H, W] (optional)
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-6
        pred = torch.clamp(pred, min=eps)
        target = torch.clamp(target, min=eps)
        
        # Compute log difference
        log_diff = torch.log(pred) - torch.log(target)
        
        if mask is not None:
            log_diff = log_diff * mask
            n_valid = mask.sum()
        else:
            n_valid = log_diff.numel()
        
        if n_valid == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Scale-invariant terms
        term1 = (log_diff ** 2).sum() / n_valid
        term2 = (self.alpha / (n_valid ** 2)) * (log_diff.sum() ** 2)
        
        loss = term1 - term2
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * n_valid
        else:
            return loss


class MobileDepthLoss(nn.Module):
    """
    Combined loss function for MobileDepth training
    Combines L1, SSIM, gradient, and scale-invariant losses
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.85,
        gradient_weight: float = 0.5,
        scale_inv_weight: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Args:
            l1_weight: Weight for L1 loss
            ssim_weight: Weight for SSIM loss
            gradient_weight: Weight for gradient loss
            scale_inv_weight: Weight for scale-invariant loss
            reduction: Reduction method for losses
        """
        super().__init__()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        self.scale_inv_weight = scale_inv_weight
        
        # Initialize loss functions
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.ssim_loss = SSIMLoss(reduction=reduction)
        self.gradient_loss = GradientLoss(reduction=reduction)
        self.scale_inv_loss = ScaleInvariantLogLoss(reduction=reduction)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: Predicted depth map [B, C, H, W]
            target: Ground truth depth map [B, C, H, W]
            mask: Valid pixel mask [B, C, H, W] (optional)
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        
        # Apply mask if provided
        if mask is not None:
            pred_masked = pred * mask
            target_masked = target * mask
        else:
            pred_masked = pred
            target_masked = target
        
        # Compute individual losses
        l1 = self.l1_loss(pred_masked, target_masked)
        ssim = self.ssim_loss(pred_masked, target_masked)
        gradient = self.gradient_loss(pred_masked, target_masked)
        scale_inv = self.scale_inv_loss(pred, target, mask)
        
        # Combine losses
        total_loss = (
            self.l1_weight * l1 +
            self.ssim_weight * ssim +
            self.gradient_weight * gradient +
            self.scale_inv_weight * scale_inv
        )
        
        # Create loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'l1_loss': l1.item(),
            'ssim_loss': ssim.item(),
            'gradient_loss': gradient.item(),
            'scale_inv_loss': scale_inv.item()
        }
        
        return total_loss, loss_dict


class MaskAwareLoss(nn.Module):
    """
    Mask-aware wrapper for handling invalid depth pixels
    """
    
    def __init__(self, base_loss: nn.Module, min_depth: float = 0.1, max_depth: float = 80.0):
        """
        Args:
            base_loss: Base loss function to wrap
            min_depth: Minimum valid depth value
            max_depth: Maximum valid depth value
        """
        super().__init__()
        self.base_loss = base_loss
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    def create_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Create mask for valid depth pixels"""
        mask = (target > self.min_depth) & (target < self.max_depth)
        return mask.float()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply mask-aware loss computation"""
        mask = self.create_mask(target)
        
        if hasattr(self.base_loss, 'forward') and 'mask' in self.base_loss.forward.__code__.co_varnames:
            return self.base_loss(pred, target, mask)
        else:
            # Apply mask manually
            pred_masked = pred * mask
            target_masked = target * mask
            return self.base_loss(pred_masked, target_masked)


# Evaluation metrics
class DepthMetrics:
    """
    Standard depth estimation evaluation metrics
    """
    
    @staticmethod
    def compute_metrics(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute standard depth estimation metrics
        
        Args:
            pred: Predicted depth [B, C, H, W]
            target: Ground truth depth [B, C, H, W]
            mask: Valid pixel mask [B, C, H, W]
        
        Returns:
            Dictionary of metrics
        """
        
        if mask is not None:
            pred = pred[mask > 0]
            target = target[mask > 0]
        else:
            pred = pred.flatten()
            target = target.flatten()
        
        if len(pred) == 0:
            return {}
        
        # Threshold metrics
        thresh = torch.max(pred / target, target / pred)
        delta1 = (thresh < 1.25).float().mean()
        delta2 = (thresh < 1.25 ** 2).float().mean()
        delta3 = (thresh < 1.25 ** 3).float().mean()
        
        # Error metrics
        abs_rel = torch.mean(torch.abs(pred - target) / target)
        sq_rel = torch.mean((pred - target) ** 2 / target)
        
        rmse = torch.sqrt(torch.mean((pred - target) ** 2))
        rmse_log = torch.sqrt(torch.mean((torch.log(pred) - torch.log(target)) ** 2))
        
        mae = torch.mean(torch.abs(pred - target))
        
        return {
            'delta1': delta1.item(),
            'delta2': delta2.item(),
            'delta3': delta3.item(),
            'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(),
            'rmse': rmse.item(),
            'rmse_log': rmse_log.item(),
            'mae': mae.item()
        }


if __name__ == "__main__":
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample data
    batch_size, channels, height, width = 4, 1, 224, 224
    pred = torch.randn(batch_size, channels, height, width, device=device, requires_grad=True)
    target = torch.randn(batch_size, channels, height, width, device=device)
    
    # Test combined loss
    loss_fn = MobileDepthLoss()
    total_loss, loss_dict = loss_fn(pred, target)
    
    print("Loss components:")
    for name, value in loss_dict.items():
        print(f"{name}: {value:.4f}")
    
    # Test metrics
    metrics = DepthMetrics.compute_metrics(pred.detach(), target)
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")