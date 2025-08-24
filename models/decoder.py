import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ChannelShuffle(nn.Module):
    """Channel shuffle operation for efficient information mixing"""
    
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        channels_per_group = channels // self.groups
        
        # Reshape and transpose
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, channels, height, width)
        
        return x


class SplitConcatenateShuffle(nn.Module):
    """
    Split-Concatenate Shuffle block for efficient upsampling
    Core component of MobileDepth decoder
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        groups: int = 2
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.groups = groups
        
        # Split channels
        self.split_channels = in_channels // 2
        
        # Branch 1: Direct upsampling
        self.branch1 = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(self.split_channels, out_channels // 2, 1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: Convolution then upsampling
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.split_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(self.split_channels, out_channels // 2, 1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Channel shuffle
        self.shuffle = ChannelShuffle(groups)
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Split input channels
        x1, x2 = torch.split(x, self.split_channels, dim=1)
        
        # Process through branches
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        
        # Concatenate
        out = torch.cat([out1, out2], dim=1)
        
        # Channel shuffle
        out = self.shuffle(out)
        
        # Final convolution
        out = self.final_conv(out)
        
        return out


class FeatureFusion(nn.Module):
    """Feature fusion module for combining encoder features with decoder features"""
    
    def __init__(self, encoder_channels: int, decoder_channels: int, out_channels: int):
        super().__init__()
        
        # Adapt encoder features to match decoder channels
        self.encoder_adapt = nn.Sequential(
            nn.Conv2d(encoder_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Adapt decoder features
        self.decoder_adapt = nn.Sequential(
            nn.Conv2d(decoder_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion convolution
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, encoder_feat, decoder_feat):
        # Adapt features
        enc_adapted = self.encoder_adapt(encoder_feat)
        dec_adapted = self.decoder_adapt(decoder_feat)
        
        # Resize encoder features to match decoder if needed
        if enc_adapted.shape[-2:] != dec_adapted.shape[-2:]:
            enc_adapted = F.interpolate(
                enc_adapted, size=dec_adapted.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        # Concatenate and fuse
        fused = torch.cat([enc_adapted, dec_adapted], dim=1)
        out = self.fusion(fused)
        
        return out


class MobileDepthDecoder(nn.Module):
    """
    MobileDepth decoder with Split-Concatenate Shuffle blocks
    """
    
    def __init__(
        self,
        encoder_channels: List[int] = [32, 48, 120, 336],
        decoder_channels: List[int] = [256, 128, 64, 32],
        output_channels: int = 1
    ):
        super().__init__()
        
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        
        # Initial processing of the deepest encoder feature
        self.initial_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Split-Concatenate Shuffle blocks
        self.scs_blocks = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        
        for i in range(len(decoder_channels) - 1):
            # SCS upsampling block
            scs_block = SplitConcatenateShuffle(
                decoder_channels[i],
                decoder_channels[i + 1],
                scale_factor=2
            )
            self.scs_blocks.append(scs_block)
            
            # Feature fusion block (skip connections)
            if i < len(encoder_channels) - 1:
                fusion_block = FeatureFusion(
                    encoder_channels[-(i + 2)],  # Corresponding encoder stage
                    decoder_channels[i + 1],
                    decoder_channels[i + 1]
                )
                self.fusion_blocks.append(fusion_block)
        
        # Final output layers
        self.output_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, 3, padding=1),
            nn.ReLU(inplace=True)  # Ensure positive depth values
        )
    
    def forward(self, encoder_features: List[torch.Tensor]):
        """
        Args:
            encoder_features: List of feature maps from encoder [stem, stage1, stage2, stage3, stage4]
        """
        # Start with the deepest feature
        x = self.initial_conv(encoder_features[-1])
        
        # Progressive upsampling with skip connections
        for i, scs_block in enumerate(self.scs_blocks):
            # Upsample
            x = scs_block(x)
            
            # Fuse with encoder features (skip connections)
            if i < len(self.fusion_blocks):
                encoder_idx = len(encoder_features) - 2 - i
                x = self.fusion_blocks[i](encoder_features[encoder_idx], x)
        
        # Final output
        depth = self.output_conv(x)
        
        return depth


class MobileDepthNet(nn.Module):
    """Complete MobileDepth network combining encoder and decoder"""
    
    def __init__(self, input_channels: int = 3, output_channels: int = 1):
        super().__init__()
        
        # Import encoder here to avoid circular imports
        from .regnet import RegNetY06
        
        self.encoder = RegNetY06(input_channels)
        self.decoder = MobileDepthDecoder(
            encoder_channels=self.encoder.stage_out_channels,
            output_channels=output_channels
        )
    
    def forward(self, x):
        # Encoder
        encoder_features = self.encoder(x)
        
        # Decoder
        depth = self.decoder(encoder_features)
        
        # Upsample to original resolution if needed
        if depth.shape[-2:] != x.shape[-2:]:
            depth = F.interpolate(
                depth, size=x.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        return depth


if __name__ == "__main__":
    # Test the decoder
    model = MobileDepthNet()
    x = torch.randn(1, 3, 224, 224)
    depth = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output depth shape: {depth.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")