import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../VMamba'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcls.models.builder import BACKBONES

# Import VMamba modules
from VMamba.vmamba import VSSM, Backbone_VSSM


@BACKBONES.register_module()
class VMambaBackbone(BaseModule):
    """VMamba backbone for FSCIL.
    
    This backbone integrates VMamba (Vision Mamba) as the feature extractor
    for Few-Shot Class-Incremental Learning, replacing traditional CNN backbones
    like ResNet with efficient State Space Models.
    
    Args:
        model_name (str): VMamba model variant name. Options: 
            - 'vmamba_tiny_s2l5': Tiny model with 2,2,5,2 depths
            - 'vmamba_small_s2l15': Small model with 2,2,15,2 depths  
            - 'vmamba_base_s2l15': Base model with 2,2,15,2 depths
            - 'vmamba_tiny_s1l8': Tiny model with ssm_ratio=1.0
            - 'vmamba_small_s1l20': Small model with 2,2,20,2 depths
            - 'vmamba_base_s1l20': Base model with 2,2,20,2 depths
        pretrained_path (str): Path to pretrained VMamba checkpoint
        out_indices (tuple): Output indices for multi-scale features
        frozen_stages (int): Number of frozen stages (-1 means not freezing any)
        channel_first (bool): Whether to use channel-first format
        init_cfg (dict, optional): Initialization config
    """
    
    # Model configurations for different VMamba variants
    MODEL_CONFIGS = {
        'vmamba_tiny_s2l5': {
            'depths': [2, 2, 5, 2], 
            'dims': [96, 192, 384, 768],
            'drop_path_rate': 0.2,
            'ssm_d_state': 1,
            'ssm_ratio': 2.0,
            'forward_type': "v05_noz",
            'mlp_ratio': 4.0,
            'downsample_version': "v3",
            'patchembed_version': "v2",
        },
        'vmamba_small_s2l15': {
            'depths': [2, 2, 15, 2], 
            'dims': [96, 192, 384, 768],
            'drop_path_rate': 0.3,
            'ssm_d_state': 1,
            'ssm_ratio': 2.0,
            'forward_type': "v05_noz",
            'mlp_ratio': 4.0,
            'downsample_version': "v3",
            'patchembed_version': "v2",
        },
        'vmamba_base_s2l15': {
            'depths': [2, 2, 15, 2], 
            'dims': [128, 256, 512, 1024],
            'drop_path_rate': 0.6,
            'ssm_d_state': 1,
            'ssm_ratio': 2.0,
            'forward_type': "v05_noz",
            'mlp_ratio': 4.0,
            'downsample_version': "v3",
            'patchembed_version': "v2",
        },
        'vmamba_tiny_s1l8': {
            'depths': [2, 2, 8, 2], 
            'dims': [96, 192, 384, 768],
            'drop_path_rate': 0.2,
            'ssm_d_state': 1,
            'ssm_ratio': 1.0,
            'forward_type': "v05_noz",
            'mlp_ratio': 4.0,
            'downsample_version': "v3",
            'patchembed_version': "v2",
        },
        'vmamba_small_s1l20': {
            'depths': [2, 2, 20, 2], 
            'dims': [96, 192, 384, 768],
            'drop_path_rate': 0.3,
            'ssm_d_state': 1,
            'ssm_ratio': 1.0,
            'forward_type': "v05_noz",
            'mlp_ratio': 4.0,
            'downsample_version': "v3",
            'patchembed_version': "v2",
        },
        'vmamba_base_s1l20': {
            'depths': [2, 2, 20, 2], 
            'dims': [128, 256, 512, 1024],
            'drop_path_rate': 0.6,
            'ssm_d_state': 1,
            'ssm_ratio': 1.0,
            'forward_type': "v05_noz",
            'mlp_ratio': 4.0,
            'downsample_version': "v3",
            'patchembed_version': "v2",
        }
    }
    
    def __init__(self,
                 model_name='vmamba_base_s2l15',
                 pretrained_path=None,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 channel_first=True,
                 init_cfg=None):
        super(VMambaBackbone, self).__init__(init_cfg=init_cfg)
        
        self.model_name = model_name
        self.pretrained_path = pretrained_path
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.channel_first = channel_first
        
        # Get model configuration
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model name: {model_name}. "
                           f"Available options: {list(self.MODEL_CONFIGS.keys())}")
        
        config = self.MODEL_CONFIGS[model_name]
        
        # Create VMamba backbone model
        self.vmamba = Backbone_VSSM(
            out_indices=out_indices,
            pretrained=pretrained_path,
            patch_size=4,
            in_chans=3,
            num_classes=1000,  # Will be ignored since we use as backbone
            depths=config['depths'],
            dims=config['dims'],
            drop_path_rate=config['drop_path_rate'],
            ssm_d_state=config['ssm_d_state'],
            ssm_ratio=config['ssm_ratio'],
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type=config['forward_type'],
            mlp_ratio=config['mlp_ratio'],
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            patch_norm=True,
            norm_layer=("ln2d" if channel_first else "ln"),
            downsample_version=config['downsample_version'],
            patchembed_version=config['patchembed_version'],
            use_checkpoint=False,
            posembed=False,
            imgsize=224,
        )
        
        # Store output channel dimensions for each stage
        self.out_channels = config['dims']
        
        # Freeze stages if specified
        self._freeze_stages()
        
    def _freeze_stages(self):
        """Freeze specified stages."""
        if self.frozen_stages >= 0:
            # Freeze patch embedding
            for param in self.vmamba.patch_embed.parameters():
                param.requires_grad = False
                
            # Freeze specified stages
            for i in range(self.frozen_stages + 1):
                if i < len(self.vmamba.layers):
                    layer = self.vmamba.layers[i]
                    layer.eval()
                    for param in layer.parameters():
                        param.requires_grad = False
                        
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W)
            
        Returns:
            tuple: Multi-scale feature maps from different stages
                - For out_indices=(0,1,2,3): returns (stage1, stage2, stage3, stage4)
                - Each stage output shape: (B, dims[i], H_i, W_i)
        """
        features = self.vmamba(x)
        
        # If only one output index, return as tuple for consistency
        if len(self.out_indices) == 1:
            return (features,)
        
        return tuple(features)
    
    def train(self, mode=True):
        """Set training mode."""
        super(VMambaBackbone, self).train(mode)
        self._freeze_stages()
        
    def init_weights(self):
        """Initialize weights."""
        if self.pretrained_path is not None:
            # Load pretrained weights
            print(f"Loading VMamba pretrained weights from: {self.pretrained_path}")
            try:
                checkpoint = torch.load(self.pretrained_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Load state dict with strict=False to handle backbone-only loading
                missing_keys, unexpected_keys = self.vmamba.load_state_dict(
                    state_dict, strict=False
                )
                
                print(f"Successfully loaded VMamba weights!")
                if missing_keys:
                    print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                    
            except Exception as e:
                print(f"Failed to load VMamba weights from {self.pretrained_path}: {e}")
                print("Initializing with random weights...")
        else:
            print("No pretrained path provided, using random initialization.") 