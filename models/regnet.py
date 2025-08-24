import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for RegNet"""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class RegNetBlock(nn.Module):
    """RegNet block with optional squeeze-and-excitation"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        bottleneck_ratio: float = 1.0,
        se_ratio: Optional[float] = None
    ):
        super().__init__()
        
        # Calculate bottleneck channels
        bottleneck_channels = int(round(out_channels * bottleneck_ratio))
        
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv2 = nn.Conv2d(
            bottleneck_channels, bottleneck_channels, 3,
            stride=stride, padding=1, groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(bottleneck_channels) if se_ratio else None
        
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        
        if self.se is not None:
            out = self.se(out)
        
        out = self.bn3(self.conv3(out))
        out += identity
        out = F.relu(out)
        
        return out


class RegNetStage(nn.Module):
    """RegNet stage consisting of multiple blocks"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int = 1,
        groups: int = 1,
        bottleneck_ratio: float = 1.0,
        se_ratio: Optional[float] = None
    ):
        super().__init__()
        
        # First block with potential stride
        layers = [
            RegNetBlock(
                in_channels, out_channels, stride,
                groups, bottleneck_ratio, se_ratio
            )
        ]
        
        # Remaining blocks
        for _ in range(1, depth):
            layers.append(
                RegNetBlock(
                    out_channels, out_channels, 1,
                    groups, bottleneck_ratio, se_ratio
                )
            )
        
        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.blocks(x)


class RegNetY06(nn.Module):
    """
    RegNetY-06 architecture for MobileDepth encoder
    Based on RegNetY-600MF configuration
    """
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # RegNetY-06 configuration
        stage_widths = [32, 48, 120, 336]
        stage_depths = [1, 3, 7, 4]
        group_widths = [8, 8, 8, 8]
        strides = [1, 2, 2, 2]
        bottleneck_ratio = 1.0
        se_ratio = 0.25
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Stages
        self.stages = nn.ModuleList()
        in_width = 32
        
        for i, (width, depth, group_width, stride) in enumerate(
            zip(stage_widths, stage_depths, group_widths, strides)
        ):
            groups = width // group_width
            stage = RegNetStage(
                in_width, width, depth, stride,
                groups, bottleneck_ratio, se_ratio
            )
            self.stages.append(stage)
            in_width = width
        
        # Store output channels for each stage (for FPN-like connections)
        self.stage_out_channels = [32] + stage_widths
    
    def forward(self, x):
        features = []
        
        # Stem
        x = self.stem(x)
        features.append(x)  # 1/2 resolution
        
        # Stages
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        return features


def create_regnet_y06(pretrained: bool = False):
    """Create RegNetY-06 model"""
    model = RegNetY06()
    
    if pretrained:
        # Note: You would load pretrained weights here if available
        # For now, we'll use random initialization
        print("Warning: Pretrained weights not available, using random initialization")
    
    return model


if __name__ == "__main__":
    # Test the model
    model = create_regnet_y06()
    x = torch.randn(1, 3, 224, 224)
    features = model(x)
    
    print("RegNetY-06 Feature Maps:")
    for i, feat in enumerate(features):
        print(f"Stage {i}: {feat.shape}")