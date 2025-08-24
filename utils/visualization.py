import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import seaborn as sns
from typing import List, Optional, Tuple, Union
import os
from PIL import Image


def colorize_depth_map(
    depth: Union[torch.Tensor, np.ndarray],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colormap: str = 'plasma'
) -> np.ndarray:
    """
    Convert depth map to colorized visualization
    
    Args:
        depth: Depth map [H, W] or [1, H, W]
        vmin: Minimum depth value for colormap
        vmax: Maximum depth value for colormap
        colormap: Matplotlib colormap name
    
    Returns:
        Colorized depth map [H, W, 3] in range [0, 255]
    """
    
    # Convert to numpy if needed
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    
    # Squeeze if needed
    if depth.ndim == 3:
        depth = depth.squeeze(0)
    
    # Handle invalid values
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Set colormap range
    if vmin is None:
        vmin = np.percentile(depth[depth > 0], 1) if np.any(depth > 0) else 0
    if vmax is None:
        vmax = np.percentile(depth[depth > 0], 99) if np.any(depth > 0) else 1
    
    # Normalize depth
    depth_norm = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(depth_norm)
    
    # Convert to uint8
    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return colored_uint8


def visualize_depth_prediction(
    image: Union[torch.Tensor, np.ndarray],
    pred_depth: Union[torch.Tensor, np.ndarray],
    gt_depth: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Depth Prediction",
    save_path: Optional[str] = None,
    show_metrics: bool = True
) -> plt.Figure:
    """
    Visualize RGB image with predicted and ground truth depth
    
    Args:
        image: RGB image [3, H, W] or [H, W, 3]
        pred_depth: Predicted depth [1, H, W] or [H, W]
        gt_depth: Ground truth depth [1, H, W] or [H, W] (optional)
        title: Figure title
        save_path: Path to save the figure
        show_metrics: Whether to compute and display metrics
    
    Returns:
        Matplotlib figure
    """
    
    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0).detach().cpu().numpy()
        else:
            image = image.detach().cpu().numpy()
    
    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.detach().cpu().numpy()
    
    if gt_depth is not None and isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.detach().cpu().numpy()
    
    # Normalize image
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Setup subplot layout
    ncols = 3 if gt_depth is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5))
    
    if ncols == 2:
        axes = [axes[0], axes[1]]
    
    # RGB Image
    axes[0].imshow(image)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Predicted Depth
    pred_colored = colorize_depth_map(pred_depth)
    axes[1].imshow(pred_colored)
    axes[1].set_title('Predicted Depth')
    axes[1].axis('off')
    
    # Ground Truth Depth
    if gt_depth is not None:
        gt_colored = colorize_depth_map(gt_depth)
        axes[2].imshow(gt_colored)
        axes[2].set_title('Ground Truth Depth')
        axes[2].axis('off')
        
        # Compute and display metrics
        if show_metrics:
            from loss.depth_losses import DepthMetrics
            
            pred_tensor = torch.from_numpy(pred_depth).unsqueeze(0) if pred_depth.ndim == 2 else torch.from_numpy(pred_depth)
            gt_tensor = torch.from_numpy(gt_depth).unsqueeze(0) if gt_depth.ndim == 2 else torch.from_numpy(gt_depth)
            
            metrics = DepthMetrics.compute_metrics(pred_tensor, gt_tensor)
            
            if metrics:
                metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()[:4]])
                fig.text(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_speed_estimation(
    image_sequence: List[Union[torch.Tensor, np.ndarray]],
    depth_sequence: List[Union[torch.Tensor, np.ndarray]],
    motion_vectors: np.ndarray,
    confidence: np.ndarray,
    estimated_speed: float,
    title: str = "Speed Estimation",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize speed estimation results
    
    Args:
        image_sequence: List of RGB images
        depth_sequence: List of depth maps
        motion_vectors: 3D motion vectors
        confidence: Confidence map
        estimated_speed: Estimated speed value
        title: Figure title
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert first two images and depths
    for i in range(min(2, len(image_sequence))):
        # Image
        img = image_sequence[i]
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0).detach().cpu().numpy()
            else:
                img = img.detach().cpu().numpy()
        
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Frame {i+1}')
        axes[0, i].axis('off')
        
        # Depth
        depth = depth_sequence[i]
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        
        depth_colored = colorize_depth_map(depth)
        axes[1, i].imshow(depth_colored)
        axes[1, i].set_title(f'Depth {i+1}')
        axes[1, i].axis('off')
    
    # Motion visualization
    if motion_vectors.ndim == 3:  # Dense flow
        # Show motion magnitude
        motion_mag = np.linalg.norm(motion_vectors, axis=2)
        im = axes[0, 2].imshow(motion_mag, cmap='hot')
        axes[0, 2].set_title('Motion Magnitude')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2])
    else:
        # Sparse flow - show as quiver plot
        axes[0, 2].text(0.5, 0.5, f'Sparse Flow\n{len(motion_vectors)} points', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Motion Vectors')
        axes[0, 2].axis('off')
    
    # Confidence map
    if confidence.ndim == 2:
        im = axes[1, 2].imshow(confidence, cmap='viridis')
        axes[1, 2].set_title('Confidence Map')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2])
    else:
        axes[1, 2].text(0.5, 0.5, f'Avg Confidence: {np.mean(confidence):.3f}', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Confidence')
        axes[1, 2].axis('off')
    
    # Add speed information
    speed_text = f"Estimated Speed: {estimated_speed:.2f} m/s\n({estimated_speed * 3.6:.1f} km/h)"
    fig.text(0.5, 0.02, speed_text, ha='center', fontsize=14, 
             bbox=dict(boxstyle="round", facecolor='lightblue'))
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_training_dashboard(
    train_losses: List[float],
    val_losses: List[float],
    metrics_history: List[dict],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive training dashboard
    
    Args:
        train_losses: Training losses over epochs
        val_losses: Validation losses over epochs
        metrics_history: Metrics history over epochs
        save_path: Path to save the dashboard
    
    Returns:
        Matplotlib figure
    """
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Loss curves
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = range(len(train_losses))
    ax1.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    if val_losses:
        val_epochs = range(len(val_losses))
        ax1.plot(val_epochs, val_losses, label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metrics plots
    if metrics_history and len(metrics_history) > 0:
        metric_epochs = range(len(metrics_history))
        
        # Delta1 accuracy
        ax2 = fig.add_subplot(gs[0, 2])
        delta1_values = [m.get('delta1', 0) for m in metrics_history]
        ax2.plot(metric_epochs, delta1_values, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('δ1 Accuracy')
        ax2.set_title('δ1 Accuracy')
        ax2.grid(True, alpha=0.3)
        
        # RMSE
        ax3 = fig.add_subplot(gs[0, 3])
        rmse_values = [m.get('rmse', 0) for m in metrics_history]
        ax3.plot(metric_epochs, rmse_values, 'r-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('RMSE')
        ax3.set_title('Root Mean Square Error')
        ax3.grid(True, alpha=0.3)
        
        # More metrics
        ax4 = fig.add_subplot(gs[1, 0])
        abs_rel_values = [m.get('abs_rel', 0) for m in metrics_history]
        ax4.plot(metric_epochs, abs_rel_values, 'b-', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Abs Rel')
        ax4.set_title('Absolute Relative Error')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        mae_values = [m.get('mae', 0) for m in metrics_history]
        ax5.plot(metric_epochs, mae_values, 'm-', linewidth=2)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('MAE')
        ax5.set_title('Mean Absolute Error')
        ax5.grid(True, alpha=0.3)
        
        # Threshold accuracies
        ax6 = fig.add_subplot(gs[1, 2:])
        delta1_values = [m.get('delta1', 0) for m in metrics_history]
        delta2_values = [m.get('delta2', 0) for m in metrics_history]
        delta3_values = [m.get('delta3', 0) for m in metrics_history]
        
        ax6.plot(metric_epochs, delta1_values, label='δ < 1.25', linewidth=2)
        ax6.plot(metric_epochs, delta2_values, label='δ < 1.25²', linewidth=2)
        ax6.plot(metric_epochs, delta3_values, label='δ < 1.25³', linewidth=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Accuracy')
        ax6.set_title('Threshold Accuracies')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Final metrics summary
        if metrics_history:
            final_metrics = metrics_history[-1]
            summary_text = "Final Metrics:\n"
            for key, value in final_metrics.items():
                summary_text += f"{key}: {value:.4f}\n"
            
            ax7 = fig.add_subplot(gs[2, :])
            ax7.text(0.1, 0.5, summary_text, transform=ax7.transAxes, fontsize=12,
                    verticalalignment='center', bbox=dict(boxstyle="round", facecolor='wheat'))
            ax7.axis('off')
    
    plt.suptitle('Training Dashboard', fontsize=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comparison_grid(
    images: List[Union[torch.Tensor, np.ndarray]],
    pred_depths: List[Union[torch.Tensor, np.ndarray]],
    gt_depths: List[Union[torch.Tensor, np.ndarray]],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a grid comparison of multiple samples
    
    Args:
        images: List of RGB images
        pred_depths: List of predicted depth maps
        gt_depths: List of ground truth depth maps
        titles: Optional list of titles for each sample
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    
    n_samples = len(images)
    fig, axes = plt.subplots(3, n_samples, figsize=(4*n_samples, 12))
    
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_samples):
        # Convert tensors
        img = images[i]
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0).detach().cpu().numpy()
            else:
                img = img.detach().cpu().numpy()
        
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # RGB Image
        axes[0, i].imshow(img)
        axes[0, i].set_title(titles[i] if titles else f'Sample {i+1}')
        axes[0, i].axis('off')
        
        # Predicted Depth
        pred_colored = colorize_depth_map(pred_depths[i])
        axes[1, i].imshow(pred_colored)
        if i == 0:
            axes[1, i].set_ylabel('Predicted', rotation=90, fontsize=12)
        axes[1, i].axis('off')
        
        # Ground Truth Depth
        gt_colored = colorize_depth_map(gt_depths[i])
        axes[2, i].imshow(gt_colored)
        if i == 0:
            axes[2, i].set_ylabel('Ground Truth', rotation=90, fontsize=12)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_depth_as_image(
    depth: Union[torch.Tensor, np.ndarray],
    save_path: str,
    colormap: str = 'plasma'
):
    """
    Save depth map as colored image
    
    Args:
        depth: Depth map
        save_path: Path to save the image
        colormap: Colormap name
    """
    
    colored_depth = colorize_depth_map(depth, colormap=colormap)
    
    # Save using PIL
    img = Image.fromarray(colored_depth)
    img.save(save_path)


def create_video_from_frames(
    frame_paths: List[str],
    output_path: str,
    fps: int = 10
):
    """
    Create video from sequence of image frames
    
    Args:
        frame_paths: List of paths to frame images
        output_path: Output video path
        fps: Frames per second
    """
    
    if not frame_paths:
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width, layers = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add frames
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video.write(frame)
    
    # Release video writer
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test visualization functions
    
    # Create sample data
    height, width = 224, 224
    
    # Sample RGB image
    image = np.random.rand(height, width, 3)
    
    # Sample depth maps
    pred_depth = np.random.rand(height, width) * 20 + 5  # 5-25m depth
    gt_depth = pred_depth + np.random.normal(0, 1, (height, width))
    
    # Test depth visualization
    fig = visualize_depth_prediction(image, pred_depth, gt_depth)
    plt.show()
    
    # Test depth colorization
    colored = colorize_depth_map(pred_depth)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(pred_depth, cmap='gray')
    plt.title('Original Depth')
    plt.subplot(1, 2, 2)
    plt.imshow(colored)
    plt.title('Colorized Depth')
    plt.show()
    
    print("Visualization tests completed!")