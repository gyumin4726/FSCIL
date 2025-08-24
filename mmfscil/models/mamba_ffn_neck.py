from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from rope import *
from timm.models.layers import trunc_normal_

from mmcls.models.builder import NECKS
from mmcls.utils import get_root_logger

from .mamba_ssm.modules.mamba_simple import Mamba
from .ss2d import SS2D


class MultiScaleSS2DAdapter(BaseModule):
    """SS2D-based Multi-Scale Feature Adapter.
    
    This adapter processes multi-scale features from different backbone layers
    using SS2D blocks for enhanced feature representation.
    
    Args:
        in_channels (int): Number of input channels from backbone layer.
        out_channels (int): Number of output channels (typically 1024).
        feat_size (int): Spatial size after adaptive pooling.
        d_state (int): Dimension of the hidden state in SS2D.
        dt_rank (int): Dimension rank in SS2D.
        ssm_expand_ratio (float): Expansion ratio for SS2D block.
        num_layers (int): Number of layers in the MLP projections.
        mid_channels (int, optional): Number of intermediate channels in MLP projections.
    """
    
    def __init__(self,
                 in_channels,
                 out_channels=1024,
                 feat_size=7,
                 d_state=256,
                 dt_rank=256,
                 ssm_expand_ratio=1.0,
                 num_layers=2,
                 mid_channels=None):
        super(MultiScaleSS2DAdapter, self).__init__(init_cfg=None)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_size = feat_size
        self.num_layers = num_layers
        self.mid_channels = in_channels * 2 if mid_channels is None else mid_channels
        
        # 1. Spatial size unification
        self.spatial_adapter = nn.AdaptiveAvgPool2d((feat_size, feat_size))
        
        # 2. Enhanced MLP projection (same as MambaNeck)
        self.mlp_proj = MambaNeck.build_mlp(in_channels, out_channels, self.mid_channels, num_layers, feat_size)
        
        # 3. SS2D block for sequence modeling
        directions = ('h', 'h_flip', 'v', 'v_flip')
        self.ss2d_block = SS2D(
            out_channels,
            ssm_ratio=ssm_expand_ratio,
            d_state=d_state,
            dt_rank=dt_rank,
            directions=directions,
            use_out_proj=False,
            use_out_norm=True
        )
        
        # 4. Positional embeddings for SS2D
        self.pos_embed = nn.Parameter(
            torch.zeros(1, feat_size * feat_size, out_channels)
        )
        trunc_normal_(self.pos_embed, std=.02)
        
        # 5. Final pooling
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        """Forward pass of Multi-Scale SSM Adapter.
        
        Args:
            x (Tensor): Input feature tensor (B, in_channels, H, W)
            
        Returns:
            Tensor: Output feature vector (B, out_channels)
        """
        B, C, H, W = x.shape
        
        # Step 1: Spatial size unification
        x = self.spatial_adapter(x)  # (B, C, feat_size, feat_size)
        
        # Step 2: Enhanced MLP projection (same as MambaNeck)
        x = self.mlp_proj(x)         # (B, out_channels, feat_size, feat_size)
        
        # Step 3: Prepare for SS2D processing
        x = x.permute(0, 2, 3, 1)    # (B, feat_size, feat_size, out_channels)
        x = x.view(B, self.feat_size * self.feat_size, -1)  # (B, feat_size^2, out_channels)
        
        # Add positional embeddings
        x = x + self.pos_embed       # (B, feat_size^2, out_channels)
        
        # Reshape back for SS2D
        x = x.view(B, self.feat_size, self.feat_size, -1)  # (B, feat_size, feat_size, out_channels)
        
        # Step 4: SS2D processing for sequence modeling
        x, _ = self.ss2d_block(x)    # (B, feat_size, feat_size, out_channels)
        
        # Step 5: Convert back to spatial format and pool
        x = x.permute(0, 3, 1, 2)    # (B, out_channels, feat_size, feat_size)
        x = self.final_pool(x)       # (B, out_channels, 1, 1)
        x = x.view(B, -1)            # (B, out_channels)
        
        return x


@NECKS.register_module()
class MambaNeck(BaseModule):
    """Dual selective SSM branch in Mamba-FSCIL framework.

        This module integrates our dual selective SSM branch for dynamic adaptation in few-shot
        class-incremental learning tasks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of intermediate channels in MLP projections, defaults to twice the in_channels if not specified.
            version (str): Specifies the version of the state space model; 'ssm' or 'ss2d'.
            use_residual_proj (bool): If True, adds a residual projection.
            d_state (int): Dimension of the hidden state in the SSM.
            d_rank (int, optional): Dimension rank in the SSM, if not provided, defaults to d_state.
            ssm_expand_ratio (float): Expansion ratio for the SSM block.
            num_layers (int): Number of layers in the MLP projections.
            num_layers_new (int, optional): Number of layers in the new branch MLP projections, defaults to num_layers if not specified.
            feat_size (int): Size of the input feature map.
            use_new_branch (bool): If True, uses an additional branch for incremental learning.
            loss_weight_supp (float): Loss weight for suppression term for base classes.
            loss_weight_supp_novel (float): Loss weight for suppression term for novel classes.
            loss_weight_sep (float): Loss weight for separation term during the base session.
            loss_weight_sep_new (float): Loss weight for separation term during the incremental session.
            param_avg_dim (str): Dimensions to average for computing averaged input-dependment parameter features; '0-1' or '0-3' or '0-1-3'.
            detach_residual (bool): If True, detaches the residual connections during the output computation.
            use_multi_scale_skip (bool): Whether to use multi-scale skip connections from different backbone layers.

            Note: Always uses attention-based skip connection fusion for optimal performance.
            multi_scale_channels (list): Channel dimensions for multi-scale features from backbone layers.
    """

    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 mid_channels=None,
                 version='ss2d',
                 use_residual_proj=False,
                 d_state=256,
                 d_rank=None,
                 ssm_expand_ratio=1,
                 num_layers=2,
                 num_layers_new=None,
                 feat_size=2,
                 use_new_branch=False,
                 loss_weight_supp=0.0,
                 loss_weight_supp_novel=0.0,
                 loss_weight_sep=0.0,
                 loss_weight_sep_new=0.0,
                 param_avg_dim='0-1-3',
                 detach_residual=False,
                 # Enhanced skip connection parameters
                 use_multi_scale_skip=False,
                 multi_scale_channels=[128, 256, 512]):
        super(MambaNeck, self).__init__(init_cfg=None)
        
        # Version selection
        self.version = version
        assert self.version in ['ssm', 'ss2d'], f'Invalid branch version: {self.version}. Must be "ssm" or "ss2d".'
        
        # SS2D Neck parameters
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.use_residual_proj = use_residual_proj
        self.mid_channels = in_channels * 2 if mid_channels is None else mid_channels
        self.feat_size = feat_size
        self.d_state = d_state
        self.d_rank = d_rank if d_rank is not None else d_state
        self.use_new_branch = use_new_branch
        self.num_layers = num_layers
        self.num_layers_new = self.num_layers if num_layers_new is None else num_layers_new
        self.detach_residual = detach_residual
        self.loss_weight_supp = loss_weight_supp
        self.loss_weight_supp_novel = loss_weight_supp_novel
        self.loss_weight_sep = loss_weight_sep
        self.loss_weight_sep_new = loss_weight_sep_new
        self.param_avg_dim = [int(item) for item in param_avg_dim.split('-')]
        self.logger = get_root_logger()
        
        # Enhanced skip connection parameters
        self.use_multi_scale_skip = use_multi_scale_skip
        self.multi_scale_channels = multi_scale_channels
        
        # Always use attention-based skip connections for MASC-M
        self.use_attention_skip = True
        
        # Log the effective configuration
        self.logger.info(f"MASC-M Enhanced Skip Connections: Using cross-attention fusion with {len(self.multi_scale_channels)} multi-scale layers")
        self.logger.info(f"Multi-Scale SS2D Adapters: {self.multi_scale_channels} → {out_channels} channels with SS2D processing")
        self.logger.info(f"Using {self.version.upper()} version for state space modeling")
        
        directions = ('h', 'h_flip', 'v', 'v_flip')

        # Positional embeddings for features
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.feat_size * self.feat_size, out_channels))
        trunc_normal_(self.pos_embed, std=.02)

        if self.use_new_branch:
            self.pos_embed_new = nn.Parameter(
                torch.zeros(1, self.feat_size * self.feat_size, out_channels))
            trunc_normal_(self.pos_embed_new, std=.02)

        # Enhanced MLP
        if self.num_layers == 3:
            self.mlp_proj = self.build_mlp(in_channels,
                                           out_channels,
                                           self.mid_channels,
                                           num_layers=3,
                                           feat_size=self.feat_size)
        elif self.num_layers == 2:
            self.mlp_proj = self.build_mlp(in_channels,
                                           out_channels,
                                           self.mid_channels,
                                           num_layers=2,
                                           feat_size=self.feat_size)

        # SSM/SS2D block initialization based on version
        if self.version == 'ssm':
            self.block = Mamba(out_channels,
                               expand=ssm_expand_ratio,
                               use_out_proj=False,
                               d_state=d_state,
                               dt_rank=self.d_rank)
        else:  # ss2d
            self.block = SS2D(out_channels,
                              ssm_ratio=ssm_expand_ratio,
                              d_state=d_state,
                              dt_rank=self.d_rank,
                              directions=directions,
                              use_out_proj=False,
                              use_out_norm=True)

        # Multi-scale skip connection adapters
        if self.use_multi_scale_skip:
            self.multi_scale_adapters = nn.ModuleList()
            for ch in self.multi_scale_channels:
                # SS2D 기반 Multi-Scale Adapter (same preprocessing as MambaNeck)
                adapter = MultiScaleSS2DAdapter(
                    in_channels=ch,
                    out_channels=out_channels,
                    feat_size=self.feat_size,
                    d_state=d_state,
                    dt_rank=self.d_rank,
                    ssm_expand_ratio=ssm_expand_ratio,
                    num_layers=self.num_layers,  # Use same num_layers as main MambaNeck
                    mid_channels=ch * 2  # Adaptive mid_channels based on input channels
                )
                self.multi_scale_adapters.append(adapter)
        
        # Attention-based skip connection weighting (always enabled)
        num_skip_sources = 1  # identity
        if self.use_multi_scale_skip:
            num_skip_sources += len(self.multi_scale_channels)  # layer1-3 채널
        if self.use_new_branch:
            num_skip_sources += 1
            
        # Cross-attention 방식 (모든 skip features를 고려한 상호 attention)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Query, Key, Value projection layers
        self.query_proj = nn.Linear(out_channels, out_channels)
        self.key_proj = nn.Linear(out_channels, out_channels)
        self.value_proj = nn.Linear(out_channels, out_channels)
        


        if self.use_new_branch:
            if self.num_layers_new == 3:
                self.mlp_proj_new = self.build_mlp(in_channels,
                                                   out_channels,
                                                   self.mid_channels,
                                                   num_layers=3,
                                                   feat_size=self.feat_size)
            elif self.num_layers_new == 2:
                self.mlp_proj_new = self.build_mlp(in_channels,
                                                   out_channels,
                                                   self.mid_channels,
                                                   num_layers=2,
                                                   feat_size=self.feat_size)

            # SSM/SS2D new branch 블록 초기화 based on version
            if self.version == 'ssm':
                self.block_new = Mamba(out_channels,
                                       expand=ssm_expand_ratio,
                                       use_out_proj=False,
                                       d_state=d_state,
                                       dt_rank=self.d_rank)
            else:  # ss2d
                self.block_new = SS2D(out_channels,
                                      ssm_ratio=ssm_expand_ratio,
                                      d_state=d_state,
                                      dt_rank=self.d_rank,
                                      directions=directions,
                                      use_out_proj=False,
                                      use_out_norm=True)

        if self.use_residual_proj:
            self.residual_proj = nn.Sequential(
                OrderedDict([
                    ('proj', nn.Linear(in_channels, out_channels, bias=False)),
                ]))

        self.init_weights()

    @staticmethod
    def build_mlp(in_channels, out_channels, mid_channels, num_layers, feat_size):
        """Builds the MLP projection part of the neck.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int): Number of mid-level channels.
            num_layers (int): Number of linear layers in the MLP.
            feat_size (int): Size of the input feature map.

        Returns:
            nn.Sequential: The MLP layers as a sequential module.
        """
        layers = []
        layers.append(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=1,
                      padding=0,
                      bias=True))
        layers.append(
            build_norm_layer(
                dict(type='LN'),
                [mid_channels, feat_size, feat_size])[1])
        layers.append(nn.LeakyReLU(0.1))

        if num_layers == 3:
            layers.append(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1,
                          bias=True))
            layers.append(
                build_norm_layer(
                    dict(type='LN'),
                    [mid_channels, feat_size, feat_size])[1])
            layers.append(nn.LeakyReLU(0.1))

        layers.append(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False))
        return nn.Sequential(*layers)

    def init_weights(self):
        """Enhanced initialization for skip connections."""
        if self.use_new_branch:
            with torch.no_grad():
                if self.version == 'ssm':
                    dim_proj = int(self.block_new.in_proj.weight.shape[0] / 2)
                    self.block_new.in_proj.weight.data[-dim_proj:, :].zero_()
                else:  # ss2d
                    dim_proj = int(self.block_new.in_proj.weight.shape[0] / 2)
                    self.block_new.in_proj.weight.data[-dim_proj:, :].zero_()
            self.logger.info(
                f'--MambaNeck zero_init_residual z: '
                f'(self.block_new.in_proj.weight{self.block_new.in_proj.weight.shape}), '
                f'{torch.norm(self.block_new.in_proj.weight.data[-dim_proj:, :])}'
            )
        
        # Initialize multi-scale SS2D adapters
        if self.use_multi_scale_skip:
            for i, adapter in enumerate(self.multi_scale_adapters):
                # Initialize MLP projection layers (same as MambaNeck)
                if hasattr(adapter, 'mlp_proj'):
                    # Initialize first Conv2d layer in MLP
                    first_layer = adapter.mlp_proj[0]  # First Conv2d layer
                    if isinstance(first_layer, nn.Conv2d):
                        nn.init.kaiming_normal_(first_layer.weight, mode='fan_out', nonlinearity='relu')
                
                # Initialize SS2D blocks with small weights for stability
                if hasattr(adapter, 'ss2d_block'):
                    # Initialize input projection with smaller weights
                    with torch.no_grad():
                        if hasattr(adapter.ss2d_block, 'in_proj'):
                            adapter.ss2d_block.in_proj.weight.data *= 0.1
                
                self.logger.info(f'Initialized MultiScaleSS2DAdapter {i} for channel {self.multi_scale_channels[i]} with {adapter.num_layers}-layer MLP')

    def forward(self, x, multi_scale_features=None):
        """Enhanced forward pass with multi-scale skip connections (MASC-M).

            Args:
                x (Tensor or tuple): Input tensor from backbone. 
                                   If tuple: (layer1_feat, layer2_feat, layer3_feat, layer4_feat)
                multi_scale_features (list, optional): List of features from different backbone layers
                                                       [layer1_feat, layer2_feat, layer3_feat]

            Returns:
                dict: A dictionary of outputs including processed features from main and new branches,
                      along with the combined final output.
            """
        # MASC-M: Extract multi-scale features from ResNet tuple output
        if isinstance(x, tuple):
            if multi_scale_features is None and len(x) > 1:
                # ResNet with out_indices=(0,1,2,3) returns (layer1, layer2, layer3, layer4)
                # Use layer1-3 as multi-scale features, layer4 as main input
                multi_scale_features = x[:-1]  # [layer1, layer2, layer3]
            x = x[-1]  # layer4 as main input
        
        # multi_scale_features가 없으면 오류 발생
        if self.use_multi_scale_skip and (multi_scale_features is None or len(multi_scale_features) == 0):
            raise ValueError('use_multi_scale_skip=True 인데 multi_scale_features가 없습니다. backbone에서 여러 레이어 출력을 반환하도록 설정하세요.')

        B, C, H, W = x.shape
        identity = x
        outputs = {}

        C, dts, Bs, Cs, C_new, dts_new, Bs_new, Cs_new = None, None, None, None, None, None, None, None

        if self.detach_residual:
            self.block.eval()
            self.mlp_proj.eval()

        # Prepare the identity projection for the residual connection
        if self.use_residual_proj:
            identity_proj = self.residual_proj(self.avg(identity).view(B, -1))
        else:
            identity_proj = self.avg(identity).view(B, -1)
        x = self.mlp_proj(identity).permute(0, 2, 3, 1).view(B, H * W, -1)

        # Process the input tensor through MLP projection and add positional embeddings
        x = x.view(B, H * W, -1) + self.pos_embed

        # SSM/SS2D processing based on version
        if self.version == 'ssm':
            # SSM block processing
            x_h, C_h = self.block(x, return_param=True)
            if isinstance(C_h, list):
                C_h, dts, Bs, Cs = C_h
                outputs.update({
                    'dts':
                    dts.view(dts.shape[0], 1, dts.shape[1], dts.shape[2]),
                    'Bs':
                    Bs.view(Bs.shape[0], 1, Bs.shape[1], Bs.shape[2]),
                    'Cs':
                    Cs.view(Cs.shape[0], 1, Cs.shape[1], Cs.shape[2])
                })
            # Handle horizontal and vertical symmetry by processing flipped versions.
            x_hf, C_hf = self.block(x.flip([1]), return_param=False)
            xs_v = rearrange(x, 'b (h w) d -> b (w h) d', h=H,
                             w=W).view(B, H * W, -1)
            x_v, C_v = self.block(xs_v, return_param=False)
            x_vf, C_vf = self.block(xs_v.flip([1]), return_param=False)

            x = x_h + x_hf.flip([1]) + rearrange(
                x_v, 'b (h w) d -> b (w h) d', h=H, w=W) + rearrange(
                    x_vf.flip([1]), 'b (h w) d -> b (w h) d', h=H, w=W)
            C = C_h + C_hf.flip([1]) + rearrange(
                C_v, 'b d (h w) -> b d (w h)', h=H, w=W) + rearrange(
                    C_vf.flip([1]), 'b d (h w) -> b d (w h)', h=H, w=W)
            x = self.avg(x.permute(0, 2, 1).reshape(B, -1, H, W)).view(B, -1)
        else:
            # SS2D processing
            x = x.view(B, H, W, -1)
            x, C = self.block(x, return_param=True)

            if isinstance(C, list):
                C, dts, Bs, Cs = C
                outputs.update({'dts': dts, 'Bs': Bs, 'Cs': Cs})
            x = self.avg(x.permute(0, 3, 1, 2)).view(B, -1)

        # New branch processing for incremental learning sessions, if enabled.
        if self.use_new_branch:
            x_new = self.mlp_proj_new(identity.detach()).permute(
                0, 2, 3, 1).view(B, H * W, -1)
            x_new += self.pos_embed_new
            
            if self.version == 'ssm':
                x_h_new, C_h_new = self.block_new(x_new, return_param=True)
                if isinstance(C_h_new, list):
                    C_h_new, dts_new, Bs_new, Cs_new = C_h_new
                    outputs.update({
                        'dts_new':
                        dts_new.view(dts_new.shape[0], 1, dts_new.shape[1],
                                     dts_new.shape[2]),
                        'Bs_new':
                        Bs_new.view(Bs_new.shape[0], 1, Bs_new.shape[1],
                                    Bs_new.shape[2]),
                        'Cs_new':
                        Cs_new.view(Cs_new.shape[0], 1, Cs_new.shape[1],
                                    Cs_new.shape[2])
                    })

                x_hf_new, C_hf_new = self.block_new(x_new.flip([1]),
                                                    return_param=False)
                xs_v_new = rearrange(x_new, 'b (h w) d -> b (w h) d', h=H,
                                     w=W).view(B, H * W, -1)
                x_v_new, C_v_new = self.block_new(xs_v_new, return_param=False)
                x_vf_new, C_vf_new = self.block_new(xs_v_new.flip([1]),
                                                    return_param=False)

                # Combine outputs from new branch.
                x_new = x_h_new + x_hf_new.flip([1]) + rearrange(
                    x_v_new, 'b (h w) d -> b (w h) d', h=H, w=W) + rearrange(
                        x_vf_new.flip([1]), 'b (h w) d -> b (w h) d', h=H, w=W)
                C_new = C_h_new + C_hf_new.flip([1]) + rearrange(
                    C_v_new, 'b d (h w) -> b d (w h)', h=H, w=W) + rearrange(
                        C_vf_new.flip([1]), 'b d (h w) -> b d (w h)', h=H, w=W)
                x_new = self.avg(x_new.permute(0, 2,
                                               1).reshape(B, -1, H,
                                                          W)).view(B, -1)
            else:
                # SS2D processing for new branch
                x_new = x_new.view(B, H, W, -1)
                x_new, C_new = self.block_new(x_new, return_param=True)
                if isinstance(C_new, list):
                    C_new, dts_new, Bs_new, Cs_new = C_new
                    outputs.update({
                        'dts_new': dts_new,
                        'Bs_new': Bs_new,
                        'Cs_new': Cs_new
                    })
                x_new = self.avg(x_new.permute(0, 3, 1, 2)).view(B, -1)

        # Initialize final output with main feature
        final_output = x
        
        # Collect skip connections
        skip_features = [identity_proj]
        
        # Multi-scale skip connections
        if self.use_multi_scale_skip:
            if multi_scale_features is not None:
                # Use actual multi-scale features when available
                for i, feat in enumerate(multi_scale_features):
                    if i < len(self.multi_scale_adapters):
                        # SS2D-based adapter processing
                        adapted_feat = self.multi_scale_adapters[i](feat)  # (B, 1024)
                        skip_features.append(adapted_feat)
                        
                        # Log adapter usage for debugging
                        if hasattr(self, 'logger') and torch.rand(1).item() < 0.01:  # 1% 확률로 로그
                            self.logger.info(f"SS2D Adapter {i}: {feat.shape} → {adapted_feat.shape}")

        # Add new branch features to skip connections
        if self.use_new_branch and 'x_new' in locals():
            # 직접 x에 x_new를 더해줌
            x = x + x_new
            # skip connection에도 추가
            skip_features.append(x_new)

        # Cross-attention based skip connection fusion (MASC-M, simplified)
        if len(skip_features) > 1:
            # Stack all skip features: [B, num_features, feature_dim]
            skip_stack = torch.stack(skip_features, dim=1)  # [B, N, D]
            
            # Prepare Query (from Mamba output), Key, Value (from skip features)
            query = self.query_proj(x).unsqueeze(1)  # [B, 1, D]
            keys = self.key_proj(skip_stack)         # [B, N, D]
            values = self.value_proj(skip_stack)     # [B, N, D]
            
            # Multi-head cross-attention
            attended_features, attention_weights = self.cross_attention(query, keys, values)
            # attention_weights: [B, 1, N]
            
            # softmax 정규화 (안정성 보강)
            weights = torch.softmax(attention_weights.squeeze(1), dim=-1)  # [B, N]
            
            # Skip features에 가중치 적용
            skip_stack = torch.stack(skip_features, dim=1)  # [B, N, D]
            weighted_skip = (weights.unsqueeze(-1) * skip_stack).sum(dim=1)  # [B, D]
            
            # 최종 출력
            final_output = x + 0.1 * weighted_skip
            
            # 디버깅: cross-attention weights 출력
            if hasattr(self, 'logger') and torch.rand(1).item() < 0.01:  # 1% 확률로 로그
                weight_values = weights[0].detach().cpu().numpy()
                # Generate dynamic feature names based on actual skip features
                feature_names = ['layer4(identity)']
                if self.use_multi_scale_skip:
                    feature_names.extend([f'layer{i+1}' for i in range(len(self.multi_scale_channels))])
                if self.use_new_branch:
                    feature_names.append('new_branch')
                feature_names = feature_names[:len(skip_features)]
                weight_info = ', '.join([f"{name}: {val:.3f}" for name, val in zip(feature_names, weight_values)])
                self.logger.info(f"Cross-attention weights: {weight_info}")


        # Store outputs
        if not self.use_new_branch:
            outputs['main'] = C if C is not None else x
            outputs['residual'] = identity_proj
        else:
            outputs['main'] = locals().get('C_new', locals().get('x_new', x))
            outputs['residual'] = x + identity_proj

        outputs['out'] = final_output
        if self.use_multi_scale_skip or self.use_attention_skip:
            outputs['skip_features'] = skip_features  # For analysis

        return outputs