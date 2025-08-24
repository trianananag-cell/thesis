import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpeedEstimationConfig:
    """Configuration for speed estimation"""
    focal_length: float = 721.5377  # KITTI camera focal length (pixels)
    baseline: float = 0.54  # KITTI stereo baseline (meters)
    time_interval: float = 0.1  # Time between frames (seconds)
    min_depth: float = 0.5  # Minimum depth for speed calculation (meters)
    max_depth: float = 50.0  # Maximum depth for speed calculation (meters)
    optical_flow_method: str = 'farneback'  # 'farneback' or 'lk'
    confidence_threshold: float = 0.5  # Confidence threshold for speed estimates


class OpticalFlowEstimator:
    """Optical flow estimation for motion analysis"""
    
    def __init__(self, method: str = 'farneback'):
        self.method = method.lower()
        
        # Farneback optical flow parameters
        self.farneback_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        # Lucas-Kanade parameters
        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
    
    def estimate_flow(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        points: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Estimate optical flow between two frames
        
        Args:
            img1: First frame (grayscale)
            img2: Second frame (grayscale)
            points: Feature points for Lucas-Kanade (optional)
        
        Returns:
            flow: Optical flow field or tracked points
            status: Status array for Lucas-Kanade (None for Farneback)
        """
        
        if self.method == 'farneback':
            flow = cv2.calcOpticalFlowPyrLK(img1, img2, **self.farneback_params)
            return flow, None
        
        elif self.method == 'lk':
            if points is None:
                # Detect corner points
                points = cv2.goodFeaturesToTrack(
                    img1,
                    maxCorners=1000,
                    qualityLevel=0.01,
                    minDistance=10,
                    blockSize=3
                )
            
            if points is not None:
                new_points, status, error = cv2.calcOpticalFlowPyrLK(
                    img1, img2, points, None, **self.lk_params
                )
                return new_points, status
            else:
                return np.array([]), np.array([])
        
        else:
            raise ValueError(f"Unknown optical flow method: {self.method}")


class DepthMotionAnalyzer:
    """Analyze motion using depth information and optical flow"""
    
    def __init__(self, config: SpeedEstimationConfig):
        self.config = config
        self.flow_estimator = OpticalFlowEstimator(config.optical_flow_method)
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array"""
        if tensor.dim() == 4:  # [B, C, H, W]
            tensor = tensor.squeeze(0).squeeze(0)  # [H, W]
        elif tensor.dim() == 3:  # [C, H, W]
            tensor = tensor.squeeze(0)  # [H, W]
        
        return tensor.detach().cpu().numpy()
    
    def estimate_3d_motion(
        self,
        depth1: torch.Tensor,
        depth2: torch.Tensor,
        image1: torch.Tensor,
        image2: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Estimate 3D motion from depth maps and images
        
        Args:
            depth1: First depth map [1, H, W] or [H, W]
            depth2: Second depth map [1, H, W] or [H, W]
            image1: First image [3, H, W]
            image2: Second image [3, H, W]
        
        Returns:
            motion_vectors: 3D motion vectors
            confidence: Confidence scores
            avg_speed: Average estimated speed
        """
        
        # Convert to numpy
        depth1_np = self.tensor_to_numpy(depth1)
        depth2_np = self.tensor_to_numpy(depth2)
        
        # Convert images to grayscale
        if image1.dim() == 3 and image1.shape[0] == 3:
            img1_gray = cv2.cvtColor(
                self.tensor_to_numpy(image1).transpose(1, 2, 0),
                cv2.COLOR_RGB2GRAY
            )
            img2_gray = cv2.cvtColor(
                self.tensor_to_numpy(image2).transpose(1, 2, 0),
                cv2.COLOR_RGB2GRAY
            )
        else:
            img1_gray = self.tensor_to_numpy(image1)
            img2_gray = self.tensor_to_numpy(image2)
        
        # Ensure uint8 format
        img1_gray = (img1_gray * 255).astype(np.uint8)
        img2_gray = (img2_gray * 255).astype(np.uint8)
        
        # Estimate optical flow
        if self.config.optical_flow_method == 'farneback':
            flow = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, **self.flow_estimator.farneback_params)
            motion_vectors, confidence = self._process_dense_flow(flow, depth1_np, depth2_np)
        else:
            points, status = self.flow_estimator.estimate_flow(img1_gray, img2_gray)
            motion_vectors, confidence = self._process_sparse_flow(points, status, depth1_np, depth2_np)
        
        # Calculate average speed
        avg_speed = self._calculate_average_speed(motion_vectors, confidence)
        
        return motion_vectors, confidence, avg_speed
    
    def _process_dense_flow(
        self,
        flow: np.ndarray,
        depth1: np.ndarray,
        depth2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process dense optical flow with depth information"""
        
        h, w = flow.shape[:2]
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Get flow vectors
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        
        # Filter valid depths
        valid_mask = (
            (depth1 > self.config.min_depth) &
            (depth1 < self.config.max_depth) &
            (depth2 > self.config.min_depth) &
            (depth2 < self.config.max_depth)
        )
        
        # Calculate 3D motion
        motion_vectors = np.zeros((h, w, 3))
        confidence = np.zeros((h, w))
        
        valid_indices = np.where(valid_mask)
        
        for i, (y, x) in enumerate(zip(valid_indices[0], valid_indices[1])):
            # Convert pixel motion to 3D motion
            dx = flow_x[y, x]
            dy = flow_y[y, x]
            
            # Calculate 3D displacement using depth information
            z1 = depth1[y, x]
            z2 = depth2[y, x]
            
            # Convert pixel coordinates to camera coordinates
            x_cam1 = (x - w/2) * z1 / self.config.focal_length
            y_cam1 = (y - h/2) * z1 / self.config.focal_length
            
            x_cam2 = (x + dx - w/2) * z2 / self.config.focal_length
            y_cam2 = (y + dy - h/2) * z2 / self.config.focal_length
            
            # 3D motion vector
            motion_3d = np.array([
                x_cam2 - x_cam1,
                y_cam2 - y_cam1,
                z2 - z1
            ])
            
            motion_vectors[y, x] = motion_3d
            
            # Simple confidence based on flow magnitude and depth consistency
            flow_mag = np.sqrt(dx**2 + dy**2)
            depth_consistency = 1.0 / (1.0 + abs(z2 - z1))
            confidence[y, x] = min(1.0, flow_mag * depth_consistency)
        
        return motion_vectors, confidence
    
    def _process_sparse_flow(
        self,
        points: np.ndarray,
        status: np.ndarray,
        depth1: np.ndarray,
        depth2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process sparse optical flow points with depth information"""
        
        if len(points) == 0:
            return np.array([]), np.array([])
        
        motion_vectors = []
        confidence_scores = []
        
        for i, (point, stat) in enumerate(zip(points, status)):
            if stat == 0:  # Invalid tracking
                continue
            
            x, y = int(point[0]), int(point[1])
            
            # Check bounds
            if x < 0 or x >= depth1.shape[1] or y < 0 or y >= depth1.shape[0]:
                continue
            
            z1 = depth1[y, x]
            z2 = depth2[y, x]
            
            # Check valid depth
            if (z1 < self.config.min_depth or z1 > self.config.max_depth or
                z2 < self.config.min_depth or z2 > self.config.max_depth):
                continue
            
            # Calculate 3D motion (simplified for sparse points)
            motion_3d = np.array([0, 0, z2 - z1])  # Primarily depth change
            motion_vectors.append(motion_3d)
            
            # Confidence based on depth consistency
            depth_consistency = 1.0 / (1.0 + abs(z2 - z1))
            confidence_scores.append(depth_consistency)
        
        return np.array(motion_vectors), np.array(confidence_scores)
    
    def _calculate_average_speed(
        self,
        motion_vectors: np.ndarray,
        confidence: np.ndarray
    ) -> float:
        """Calculate average speed from motion vectors"""
        
        if len(motion_vectors) == 0:
            return 0.0
        
        # Calculate speed magnitudes
        if motion_vectors.ndim == 3:  # Dense flow
            speed_map = np.linalg.norm(motion_vectors, axis=2)
            valid_mask = confidence > self.config.confidence_threshold
            
            if np.any(valid_mask):
                weights = confidence[valid_mask]
                speeds = speed_map[valid_mask]
                avg_speed = np.average(speeds, weights=weights) / self.config.time_interval
            else:
                avg_speed = 0.0
        else:  # Sparse flow
            speeds = np.linalg.norm(motion_vectors, axis=1)
            valid_mask = confidence > self.config.confidence_threshold
            
            if np.any(valid_mask):
                weights = confidence[valid_mask]
                speeds = speeds[valid_mask]
                avg_speed = np.average(speeds, weights=weights) / self.config.time_interval
            else:
                avg_speed = 0.0
        
        return float(avg_speed)


class SpeedEstimator(nn.Module):
    """
    Complete speed estimation system using MobileDepth predictions
    """
    
    def __init__(self, config: SpeedEstimationConfig = None):
        super().__init__()
        
        self.config = config or SpeedEstimationConfig()
        self.motion_analyzer = DepthMotionAnalyzer(self.config)
        
        # Optional: Neural network for speed refinement
        self.speed_refiner = nn.Sequential(
            nn.Linear(7, 32),  # Input: [avg_speed, depth_stats(4), flow_stats(2)]
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Output: refined speed
            nn.ReLU()  # Ensure positive speed
        )
    
    def forward(
        self,
        depth_sequence: List[torch.Tensor],
        image_sequence: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Estimate speed from depth and image sequences
        
        Args:
            depth_sequence: List of depth maps [B, 1, H, W]
            image_sequence: List of images [B, 3, H, W]
        
        Returns:
            speed_estimates: Estimated speeds [B]
            details: Dictionary with detailed analysis
        """
        
        batch_size = depth_sequence[0].shape[0]
        speed_estimates = []
        all_details = []
        
        for b in range(batch_size):
            # Extract batch sample
            depth1 = depth_sequence[0][b]  # [1, H, W]
            depth2 = depth_sequence[1][b]  # [1, H, W]
            image1 = image_sequence[0][b]  # [3, H, W]
            image2 = image_sequence[1][b]  # [3, H, W]
            
            # Estimate motion
            motion_vectors, confidence, avg_speed = self.motion_analyzer.estimate_3d_motion(
                depth1, depth2, image1, image2
            )
            
            # Calculate additional features for refinement
            depth_stats = self._calculate_depth_statistics(depth1, depth2)
            flow_stats = self._calculate_flow_statistics(motion_vectors, confidence)
            
            # Combine features
            features = torch.tensor([
                avg_speed,
                *depth_stats,
                *flow_stats
            ], dtype=torch.float32, device=depth1.device)
            
            # Refine speed estimate
            refined_speed = self.speed_refiner(features.unsqueeze(0)).squeeze(0)
            speed_estimates.append(refined_speed)
            
            # Store details
            details = {
                'raw_speed': avg_speed,
                'refined_speed': refined_speed.item(),
                'motion_vectors': motion_vectors,
                'confidence': confidence,
                'depth_stats': depth_stats,
                'flow_stats': flow_stats
            }
            all_details.append(details)
        
        speed_estimates = torch.stack(speed_estimates)
        
        # Aggregate details for batch
        batch_details = {
            'raw_speeds': [d['raw_speed'] for d in all_details],
            'refined_speeds': [d['refined_speed'] for d in all_details],
            'mean_confidence': np.mean([np.mean(d['confidence']) for d in all_details if len(d['confidence']) > 0])
        }
        
        return speed_estimates, batch_details
    
    def _calculate_depth_statistics(
        self,
        depth1: torch.Tensor,
        depth2: torch.Tensor
    ) -> List[float]:
        """Calculate depth-related statistics"""
        
        depth1_np = self.motion_analyzer.tensor_to_numpy(depth1)
        depth2_np = self.motion_analyzer.tensor_to_numpy(depth2)
        
        # Valid depth masks
        valid1 = (depth1_np > self.config.min_depth) & (depth1_np < self.config.max_depth)
        valid2 = (depth2_np > self.config.min_depth) & (depth2_np < self.config.max_depth)
        
        if np.any(valid1) and np.any(valid2):
            mean_depth1 = np.mean(depth1_np[valid1])
            mean_depth2 = np.mean(depth2_np[valid2])
            depth_change = mean_depth2 - mean_depth1
            depth_variance = np.var(depth1_np[valid1])
        else:
            mean_depth1 = 0.0
            mean_depth2 = 0.0
            depth_change = 0.0
            depth_variance = 0.0
        
        return [mean_depth1, mean_depth2, depth_change, depth_variance]
    
    def _calculate_flow_statistics(
        self,
        motion_vectors: np.ndarray,
        confidence: np.ndarray
    ) -> List[float]:
        """Calculate optical flow statistics"""
        
        if len(motion_vectors) == 0:
            return [0.0, 0.0]
        
        if motion_vectors.ndim == 3:  # Dense flow
            flow_magnitude = np.linalg.norm(motion_vectors, axis=2)
            valid_mask = confidence > self.config.confidence_threshold
            
            if np.any(valid_mask):
                mean_flow = np.mean(flow_magnitude[valid_mask])
                flow_variance = np.var(flow_magnitude[valid_mask])
            else:
                mean_flow = 0.0
                flow_variance = 0.0
        else:  # Sparse flow
            flow_magnitude = np.linalg.norm(motion_vectors, axis=1)
            valid_mask = confidence > self.config.confidence_threshold
            
            if np.any(valid_mask):
                mean_flow = np.mean(flow_magnitude[valid_mask])
                flow_variance = np.var(flow_magnitude[valid_mask])
            else:
                mean_flow = 0.0
                flow_variance = 0.0
        
        return [mean_flow, flow_variance]


# Utility functions for speed estimation
def create_speed_estimator(config: Optional[SpeedEstimationConfig] = None) -> SpeedEstimator:
    """Create a speed estimator with default or custom configuration"""
    return SpeedEstimator(config)


def estimate_speed_from_depths(
    depth1: torch.Tensor,
    depth2: torch.Tensor,
    time_interval: float = 0.1,
    focal_length: float = 721.5377
) -> float:
    """
    Simple utility function to estimate speed from two depth maps
    
    Args:
        depth1: First depth map
        depth2: Second depth map
        time_interval: Time between frames
        focal_length: Camera focal length
    
    Returns:
        Estimated speed in m/s
    """
    
    config = SpeedEstimationConfig(
        focal_length=focal_length,
        time_interval=time_interval
    )
    
    analyzer = DepthMotionAnalyzer(config)
    
    # Create dummy images (motion estimation will be depth-based only)
    h, w = depth1.shape[-2:]
    dummy_img = torch.zeros(3, h, w)
    
    _, _, speed = analyzer.estimate_3d_motion(depth1, depth2, dummy_img, dummy_img)
    
    return speed


if __name__ == "__main__":
    # Test speed estimation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample data
    batch_size = 2
    height, width = 224, 224
    
    depth_sequence = [
        torch.rand(batch_size, 1, height, width, device=device) * 20 + 5,  # 5-25m depth
        torch.rand(batch_size, 1, height, width, device=device) * 20 + 5
    ]
    
    image_sequence = [
        torch.rand(batch_size, 3, height, width, device=device),
        torch.rand(batch_size, 3, height, width, device=device)
    ]
    
    # Test speed estimator
    speed_estimator = SpeedEstimator()
    speed_estimates, details = speed_estimator(depth_sequence, image_sequence)
    
    print(f"Speed estimates: {speed_estimates}")
    print(f"Details: {details}")