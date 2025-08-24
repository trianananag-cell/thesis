import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional, List
import pandas as pd


class KITTIDepthDataset(Dataset):
    """
    KITTI Depth Prediction Dataset loader for the Kaggle dataset
    https://www.kaggle.com/datasets/artemmmtry/kitti-depth-prediction-evaluation
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (224, 224),
        depth_scale: float = 256.0,
        max_depth: float = 80.0,
        transforms: Optional[A.Compose] = None
    ):
        """
        Args:
            data_dir: Path to the dataset directory
            split: 'train' or 'val'
            image_size: Target image size (height, width)
            depth_scale: Scale factor for depth values (KITTI uses 256)
            max_depth: Maximum depth value to consider
            transforms: Albumentations transforms
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        
        # Setup paths
        self.image_dir = os.path.join(data_dir, split, 'image')
        self.depth_dir = os.path.join(data_dir, split, 'groundtruth')
        
        # Get file lists
        self.image_files = self._get_file_list(self.image_dir, '.png')
        self.depth_files = self._get_file_list(self.depth_dir, '.png')
        
        # Ensure matching files
        self._verify_file_pairs()
        
        # Default transforms if none provided
        if transforms is None:
            self.transforms = self._get_default_transforms()
        else:
            self.transforms = transforms
    
    def _get_file_list(self, directory: str, extension: str) -> List[str]:
        """Get sorted list of files with given extension"""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = [f for f in os.listdir(directory) if f.endswith(extension)]
        return sorted(files)
    
    def _verify_file_pairs(self):
        """Verify that image and depth files match"""
        if len(self.image_files) != len(self.depth_files):
            raise ValueError(
                f"Mismatch between image files ({len(self.image_files)}) "
                f"and depth files ({len(self.depth_files)})"
            )
        
        # Check that filenames match (excluding extension differences)
        for img_file, depth_file in zip(self.image_files, self.depth_files):
            img_base = os.path.splitext(img_file)[0]
            depth_base = os.path.splitext(depth_file)[0]
            if img_base != depth_base:
                print(f"Warning: Filename mismatch - {img_file} vs {depth_file}")
    
    def _get_default_transforms(self) -> A.Compose:
        """Get default augmentation transforms"""
        if self.split == 'train':
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05,
                    p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'depth': 'mask'})
        else:
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'depth': 'mask'})
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load RGB image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_depth(self, depth_path: str) -> np.ndarray:
        """Load depth map"""
        # KITTI depth maps are stored as 16-bit PNG images
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Could not load depth: {depth_path}")
        
        # Convert to float and scale
        depth = depth.astype(np.float32) / self.depth_scale
        
        # Clamp depth values
        depth = np.clip(depth, 0, self.max_depth)
        
        return depth
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image and depth
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        
        image = self._load_image(image_path)
        depth = self._load_depth(depth_path)
        
        # Apply transforms
        transformed = self.transforms(image=image, depth=depth)
        image = transformed['image']
        depth = transformed['depth']
        
        # Ensure depth is float32 tensor with shape [1, H, W]
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth).float()
        
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        
        return image, depth
    
    def get_sample_paths(self, idx: int) -> Tuple[str, str]:
        """Get file paths for a sample (useful for debugging)"""
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        return image_path, depth_path


class KITTISpeedDataset(Dataset):
    """
    Extended KITTI dataset for speed estimation using consecutive frames
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sequence_length: int = 2,
        time_interval: float = 0.1,  # Time between frames in seconds
        **kwargs
    ):
        """
        Args:
            data_dir: Path to the dataset directory
            split: 'train' or 'val'
            sequence_length: Number of consecutive frames to use
            time_interval: Time interval between frames
            **kwargs: Additional arguments for KITTIDepthDataset
        """
        self.base_dataset = KITTIDepthDataset(data_dir, split, **kwargs)
        self.sequence_length = sequence_length
        self.time_interval = time_interval
        
        # Calculate valid indices for sequences
        self.valid_indices = list(range(len(self.base_dataset) - sequence_length + 1))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
        """
        Returns:
            images: List of consecutive images
            depths: List of consecutive depth maps  
            time_interval: Time interval between frames
        """
        start_idx = self.valid_indices[idx]
        
        images = []
        depths = []
        
        for i in range(self.sequence_length):
            image, depth = self.base_dataset[start_idx + i]
            images.append(image)
            depths.append(depth)
        
        return images, depths, self.time_interval


def create_kitti_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    val_split: float = 0.2,
    speed_estimation: bool = False,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for KITTI dataset
    
    Args:
        data_dir: Path to the dataset directory
        batch_size: Batch size for dataloaders
        image_size: Target image size
        num_workers: Number of worker processes
        val_split: Validation split ratio (if no separate val folder)
        speed_estimation: Whether to use speed estimation dataset
        **kwargs: Additional arguments for dataset
    
    Returns:
        train_loader, val_loader
    """
    
    # Check if separate train/val folders exist
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        # Use separate folders
        if speed_estimation:
            train_dataset = KITTISpeedDataset(data_dir, 'train', image_size=image_size, **kwargs)
            val_dataset = KITTISpeedDataset(data_dir, 'val', image_size=image_size, **kwargs)
        else:
            train_dataset = KITTIDepthDataset(data_dir, 'train', image_size=image_size, **kwargs)
            val_dataset = KITTIDepthDataset(data_dir, 'val', image_size=image_size, **kwargs)
    else:
        # Split single dataset
        full_dataset = KITTISpeedDataset(data_dir, image_size=image_size, **kwargs) if speed_estimation else KITTIDepthDataset(data_dir, image_size=image_size, **kwargs)
        
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def collate_fn_speed(batch):
    """Custom collate function for speed estimation dataset"""
    images_list, depths_list, time_intervals = zip(*batch)
    
    # Stack sequences
    batch_images = []
    batch_depths = []
    
    seq_len = len(images_list[0])
    for i in range(seq_len):
        images_at_t = torch.stack([images[i] for images in images_list])
        depths_at_t = torch.stack([depths[i] for depths in depths_list])
        batch_images.append(images_at_t)
        batch_depths.append(depths_at_t)
    
    time_intervals = torch.tensor(time_intervals)
    
    return batch_images, batch_depths, time_intervals


if __name__ == "__main__":
    # Test the dataset
    data_dir = "/kaggle/input/kitti-depth-prediction-evaluation"
    
    try:
        dataset = KITTIDepthDataset(data_dir, split='train')
        print(f"Dataset loaded successfully: {len(dataset)} samples")
        
        # Test a sample
        image, depth = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Depth shape: {depth.shape}")
        print(f"Depth range: {depth.min().item():.3f} - {depth.max().item():.3f}")
        
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        print("This is expected when running outside Kaggle environment")