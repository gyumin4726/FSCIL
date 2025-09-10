"""
MoE-FSCIL Neck: Mixture of Experts for Few-Shot Class-Incremental Learning

This module replaces the traditional branch structure with a more efficient and scalable
Mixture of Experts (MoE) architecture for FSCIL tasks.

Key innovations:
1. Branch structure elimination → Single MoE architecture
2. Dynamic expert allocation based on input characteristics
3. Constant computational cost regardless of expert count
4. FSCIL-specific gating mechanisms
"""

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
    """SS2D-based Multi-Scale Feature Adapter for MoE-FSCIL.
    
    This adapter processes multi-scale features from different backbone layers
    using SS2D blocks for enhanced feature representation, following the same
    pattern as MambaNeck.
    
    Args:
        in_channels (int): Number of input channels from backbone layer.
        out_channels (int): Number of output channels (typically 512).
        feat_size (int): Spatial size after adaptive pooling.
        d_state (int): Dimension of the hidden state in SS2D.
        dt_rank (int): Dimension rank in SS2D.
        ssm_expand_ratio (float): Expansion ratio for SS2D block.
        num_layers (int): Number of layers in the MLP projections.
        mid_channels (int, optional): Number of intermediate channels in MLP projections.
    """
    
    def __init__(self,
                 in_channels,
                 out_channels=512,
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
        self.mlp_proj = self._build_mlp(in_channels, out_channels, self.mid_channels, num_layers, feat_size)
        
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
        
    def _build_mlp(self, in_channels, out_channels, mid_channels, num_layers, feat_size):
        """Build MLP projection layers (same as MoEFSCILNeck)."""
        layers = []
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=True))
        layers.append(build_norm_layer(dict(type='LN'), [mid_channels, feat_size, feat_size])[1])
        layers.append(nn.LeakyReLU(0.1))
        
        if num_layers == 3:
            layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=True))
            layers.append(build_norm_layer(dict(type='LN'), [mid_channels, feat_size, feat_size])[1])
            layers.append(nn.LeakyReLU(0.1))
        
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass of Multi-Scale SS2D Adapter.
        
        Args:
            x (Tensor): Input feature tensor (B, in_channels, H, W)
            
        Returns:
            Tensor: Output feature vector (B, out_channels)
        """
        B, C, H, W = x.shape
        
        # Step 1: Spatial size unification
        x = self.spatial_adapter(x)  # (B, C, feat_size, feat_size)
        
        # Step 2: Enhanced MLP projection (same as MoEFSCILNeck)
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


class FSCILGate(nn.Module):
    """
    FSCIL-adapted SwitchGate based on official MoE-Mamba implementation.
    
    Follows the official SwitchGate pattern but adds FSCIL-specific features:
    - Session-aware routing (base vs incremental)
    - Load balancing for stable training
    - Capacity control for expert utilization
    """
    
    def __init__(self,
                 dim,
                 num_experts: int,
                 top_k: int = 1,
                 capacity_factor: float = 1.25,
                 epsilon: float = 1e-6,
                 use_aux_loss: bool = True):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.use_aux_loss = use_aux_loss
        
        # Simple gating network (following official implementation)
        self.w_gate = nn.Linear(dim, num_experts)
        
        # Removed session embedding for base session stability
        
    def forward(self, x: torch.Tensor, training_phase: str = 'base'):
        """
        Forward pass following official SwitchGate pattern with FSCIL adaptations.
        
        Args:
            x: Input features [B, dim] (flattened for gating)
            training_phase: 'base' or 'incremental' for session-aware routing
            
        Returns:
            gate_scores: Expert selection probabilities [B, num_experts]
            aux_loss: Load balancing auxiliary loss (if enabled)
        """
        # Compute raw gate scores directly (no session context for stability)
        raw_gate_scores = F.softmax(self.w_gate(x), dim=-1)  # [B, num_experts]
        
        # Determine capacity and apply top-k gating (configurable)
        capacity = int(self.capacity_factor * x.size(0))
        top_k_scores, top_k_indices = raw_gate_scores.topk(self.top_k, dim=-1)  # Use configurable top_k
        
        # Create sparsity mask
        mask = torch.zeros_like(raw_gate_scores).scatter_(1, top_k_indices, 1)
        masked_gate_scores = raw_gate_scores * mask
        
        # Normalize gate scores for dispatch (official pattern)
        denominators = masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        gate_scores = (masked_gate_scores / denominators) * capacity  # For dispatch only!
        
        # Compute auxiliary loss for load balancing (CORRECT Switch Transformer approach)
        aux_loss = None
        if self.use_aux_loss:
            # importance: 원본 softmax 확률의 평균 (soft routing 기준 기대 분포)
            importance = raw_gate_scores.mean(0)     # [num_experts] - RAW softmax!
            
            # load: 실제 top-1 dispatch 결과 (hard routing 분포)
            load = mask.float().mean(0)              # [num_experts] - actual selection
            
            # load balancing loss: 진짜 "soft vs hard" 분포 비교!
            # raw_gate_scores vs actual selection으로 제대로 된 load balancing
            aux_loss = 0.01 * ((load - importance) ** 2).mean()  # 가중치 0.01로 약화
        
        return gate_scores, aux_loss
    



class SS2DExpert(nn.Module):
    """
    FSCIL Expert - Pure FeedForward network (following official MoE-Mamba pattern).
    
    Key design decisions:
    1. SS2D processing is shared (before MoE) - following original MambaNeck pattern
    2. Each expert specializes in post-SS2D feature transformation
    3. Simple FFN structure like official MoE-Mamba implementation
    """
    
    def __init__(self,
                 dim,
                 expert_id=0,
                 d_state=256,
                 dt_rank=256,
                 ssm_expand_ratio=1.0):
        super().__init__()
        self.dim = dim
        self.expert_id = expert_id
        self.d_state = d_state
        self.dt_rank = dt_rank
        
        # Expert-specific SS2D block
        directions = ('h', 'h_flip', 'v', 'v_flip')
        self.ss2d_block = SS2D(
            dim,
            ssm_ratio=ssm_expand_ratio,
            d_state=d_state,
            dt_rank=dt_rank,
            directions=directions,
            use_out_proj=False,
            use_out_norm=True
        )
        
        # Expert-specific normalization
        self.norm = nn.LayerNorm(dim)
        
        # Global average pooling for spatial aggregation
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        """
        SS2D Expert forward pass - specialized sequence modeling.
        
        Args:
            x: Input features [B, H, W, dim] (spatial features before SS2D)
            
        Returns:
            output: Expert-processed features [B, dim] (flattened)
        """
        B, H, W, dim = x.shape
        
        # Expert-specific SS2D processing
        x_expert, _ = self.ss2d_block(x)  # [B, H, W, dim]
        
        # Spatial aggregation and flattening
        x_expert = x_expert.permute(0, 3, 1, 2)  # [B, dim, H, W]
        x_expert = self.avg_pool(x_expert).view(B, -1)  # [B, dim]
        
        # Normalize output
        output = self.norm(x_expert)
        
        return output


class MoEFSCIL(nn.Module):
    """
    Mixture of Experts module specifically designed for FSCIL tasks.
    
    Key features:
    - Multiple experts specializing in different aspects of FSCIL
    - FSCIL-specific gating mechanism
    - Load balancing for stable training
    - Efficient sparse activation
    """
    
    def __init__(self,
                 dim,
                 num_experts=4,
                 top_k=2,
                 feat_size=7,
                 capacity_factor=1.25,
                 d_state=256,
                 dt_rank=256,
                 ssm_expand_ratio=1.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # FSCIL-specific gating mechanism
        self.gate = FSCILGate(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor
        )
        
        # Create expert modules (pure FFN, following official MoE-Mamba)
        # SS2D experts (each with own SS2D block)
        self.experts = nn.ModuleList([
            SS2DExpert(
                dim=dim,
                expert_id=i,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_expand_ratio=ssm_expand_ratio
            )
            for i in range(num_experts)
        ])
        
    def forward(self, x, training_phase='base'):
        """
        SS2D MoE forward pass with FSCIL-specific routing.
        
        Args:
            x: Input features [B, H, W, dim] (spatial features for SS2D experts)
            training_phase: 'base' or 'incremental'
            
        Returns:
            output: Mixed expert outputs [B, dim]
            aux_loss: Load balancing loss
        """
        B, H, W, dim = x.shape
        
        # Flatten for gating decision (gate needs [B, dim] input)
        x_flat = F.adaptive_avg_pool2d(x.permute(0, 3, 1, 2), (1, 1)).view(B, -1)  # [B, dim]
        
        # Get expert selection probabilities and routing mask
        gate_scores, aux_loss = self.gate(x_flat, training_phase)  # [B, num_experts]
        
        # Find top-k experts for each token (efficient sparse routing)
        top_k_scores, top_k_indices = gate_scores.topk(self.top_k, dim=-1)  # [B, top_k]
        
        # Initialize output
        mixed_output = torch.zeros(B, dim, device=x.device, dtype=x.dtype)
        
        # Process only selected experts (sparse activation)
        for i in range(B):  # For each sample in batch
            for k in range(self.top_k):  # For each selected expert
                expert_idx = top_k_indices[i, k].item()
                expert_weight = top_k_scores[i, k]
                
                # SS2D expert processes spatial input
                expert_output = self.experts[expert_idx](x[i:i+1])  # [1, dim]
                mixed_output[i] += expert_weight * expert_output.squeeze(0)
        
        return mixed_output, aux_loss


@NECKS.register_module()
class MoEFSCILNeck(BaseModule):
    """
    MoE-FSCIL Neck: Revolutionary replacement for branch-based FSCIL architectures.
    
    This neck completely eliminates the traditional branch structure and replaces it
    with a scalable Mixture of Experts approach. Key advantages:
    
    1. Constant computational cost regardless of expert count
    2. Dynamic expert specialization for different classes/sessions
    3. Better scalability for long-term incremental learning
    4. Maintains all benefits of dynamic parameter allocation
    
    Args:
        in_channels (int): Number of input channels from backbone
        out_channels (int): Number of output channels
        num_experts (int): Number of expert modules (replaces branch count)
        top_k (int): Number of experts to activate per input
        version (str): 'ssm' or 'ss2d' for state space modeling
        d_state (int): Dimension of hidden state in SSM
        dt_rank (int): Dimension rank in SSM
        ssm_expand_ratio (float): Expansion ratio for SSM blocks
        feat_size (int): Spatial size of input features
        mid_channels (int): Intermediate channels in MLP projection
        num_layers (int): Number of MLP layers
        use_residual_proj (bool): Whether to use residual projection
        use_multi_scale_skip (bool): Whether to use multi-scale skip connections
        multi_scale_channels (list): Channel dimensions for multi-scale features
        
        # Note: Traditional FSCIL regularization losses are replaced by MoE load balancing
    """
    
    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 num_experts=4,
                 top_k=2,
                 version='ss2d',
                 d_state=256,
                 dt_rank=None,
                 ssm_expand_ratio=1.0,
                 feat_size=2,
                 mid_channels=None,
                 num_layers=2,
                 use_residual_proj=False,
                 # FSCIL suppression loss parameters
                 loss_weight_supp=0.0,
                 loss_weight_supp_novel=0.0,
                 # FSCIL separation loss parameters
                 loss_weight_sep=0.0,
                 loss_weight_sep_new=0.0,
                 # Parameter averaging for separation loss
                 param_avg_dim='0-1-3',
                 # Multi-scale skip connection parameters
                 use_multi_scale_skip=False,
                 multi_scale_channels=[128, 256, 512]):
        super(MoEFSCILNeck, self).__init__(init_cfg=None)
        
        # Core parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.version = version
        self.feat_size = feat_size
        self.mid_channels = in_channels * 2 if mid_channels is None else mid_channels
        self.num_layers = num_layers
        self.use_residual_proj = use_residual_proj
        
        # Multi-scale skip connection parameters
        self.use_multi_scale_skip = use_multi_scale_skip
        self.multi_scale_channels = multi_scale_channels
        
        # FSCIL suppression loss parameters
        self.loss_weight_supp = loss_weight_supp
        self.loss_weight_supp_novel = loss_weight_supp_novel
        
        # FSCIL separation loss parameters
        self.loss_weight_sep = loss_weight_sep
        self.loss_weight_sep_new = loss_weight_sep_new
        
        # Parameter averaging dimensions for separation loss
        self.param_avg_dim = [int(item) for item in param_avg_dim.split('-')]
        
        # Logger
        self.logger = get_root_logger()
        self.logger.info(f"MoE-FSCIL Neck initialized: {num_experts} experts, top-{top_k} activation")
        
        if self.use_multi_scale_skip:
            self.logger.info(f"Enhanced MoE with Multi-Scale Skip Connections: {len(self.multi_scale_channels)} layers")
            self.logger.info(f"Multi-Scale SS2D Adapters: {self.multi_scale_channels} → {out_channels} channels")
        
        # Global average pooling
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        
        # MLP projection (shared preprocessing - following original pattern)
        self.mlp_proj = self._build_mlp(
            in_channels, out_channels, self.mid_channels, num_layers, feat_size
        )
        
        # Shared positional embeddings (like original MambaNeck)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, feat_size * feat_size, out_channels)
        )
        trunc_normal_(self.pos_embed, std=.02)
        
        # Multi-scale skip connection adapters
        if self.use_multi_scale_skip:
            self.multi_scale_adapters = nn.ModuleList()
            for ch in self.multi_scale_channels:
                # SS2D 기반 Multi-Scale Adapter (same preprocessing as MoEFSCILNeck)
                adapter = MultiScaleSS2DAdapter(
                    in_channels=ch,
                    out_channels=out_channels,
                    feat_size=self.feat_size,
                    d_state=d_state,
                    dt_rank=dt_rank if dt_rank is not None else d_state,
                    ssm_expand_ratio=ssm_expand_ratio,
                    num_layers=self.num_layers,  # Use same num_layers as main MoEFSCILNeck
                    mid_channels=ch * 2  # Adaptive mid_channels based on input channels
                )
                self.multi_scale_adapters.append(adapter)
        
        # Cross-attention based skip connection weighting (when multi-scale is enabled)
        if self.use_multi_scale_skip:
            num_skip_sources = 1  # identity
            num_skip_sources += len(self.multi_scale_channels)  # multi-scale features
            
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
        
        # Note: No shared SS2D block - each expert has its own SS2D for specialization
        
        # SS2D MoE module (each expert has its own SS2D block)
        self.moe = MoEFSCIL(
            dim=out_channels,
            num_experts=num_experts,
            top_k=top_k,
            feat_size=feat_size,
            d_state=d_state,
            dt_rank=dt_rank if dt_rank is not None else d_state,
            ssm_expand_ratio=ssm_expand_ratio
        )
        
        # Residual projection
        if self.use_residual_proj:
            self.residual_proj = nn.Sequential(
                OrderedDict([
                    ('proj', nn.Linear(in_channels, out_channels, bias=False)),
                ])
            )
        
        self.init_weights()
        
    def _build_mlp(self, in_channels, out_channels, mid_channels, num_layers, feat_size):
        """Build MLP projection layers."""
        layers = []
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=True))
        layers.append(build_norm_layer(dict(type='LN'), [mid_channels, feat_size, feat_size])[1])
        layers.append(nn.LeakyReLU(0.1))
        
        if num_layers == 3:
            layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=True))
            layers.append(build_norm_layer(dict(type='LN'), [mid_channels, feat_size, feat_size])[1])
            layers.append(nn.LeakyReLU(0.1))
        
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False))
        return nn.Sequential(*layers)
    

    
    def init_weights(self):
        """Initialize weights with proper scaling for MoE and multi-scale adapters."""
        self.logger.info("🔧 Initializing MoE-FSCIL Neck weights...")
        
        # Initialize multi-scale SS2D adapters
        if self.use_multi_scale_skip:
            for i, adapter in enumerate(self.multi_scale_adapters):
                # Initialize MLP projection layers (same as MoEFSCILNeck)
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
    
    def forward(self, x, multi_scale_features=None, training_phase='base'):
        """
        Enhanced forward pass with MoE processing and multi-scale skip connections.
        
        Args:
            x: Input tensor [B, C, H, W] or tuple (layer1, layer2, layer3, layer4)
            multi_scale_features (list, optional): List of features from different backbone layers
                                                  [layer1_feat, layer2_feat, layer3_feat]
            training_phase: 'base' or 'incremental' for session-aware processing
            
        Returns:
            dict: Output dictionary containing:
                - 'out': Final enhanced MoE output with multi-scale fusion
                - 'aux_loss': Load balancing auxiliary loss
                - 'main': MoE output (for compatibility)
                - 'residual': Residual connection
                - 'skip_features': Skip features for analysis (if multi-scale enabled)
        """
        
        # Enhanced MoE: Extract multi-scale features from ResNet tuple output
        if isinstance(x, tuple):
            if multi_scale_features is None and len(x) > 1:
                # ResNet with out_indices=(0,1,2,3) returns (layer1, layer2, layer3, layer4)
                # Use layer1-3 as multi-scale features, layer4 as main input
                multi_scale_features = x[:-1]  # [layer1, layer2, layer3]
            x = x[-1]  # layer4 as main input
        
        # multi_scale_features가 없으면 오류 발생 (multi-scale 사용시에만)
        if self.use_multi_scale_skip and (multi_scale_features is None or len(multi_scale_features) == 0):
            raise ValueError('use_multi_scale_skip=True 인데 multi_scale_features가 없습니다. backbone에서 여러 레이어 출력을 반환하도록 설정하세요.')

        B, C, H, W = x.shape
        identity = x
        outputs = {}
        
        # Prepare residual connection
        if self.use_residual_proj:
            identity_proj = self.residual_proj(self.avg(identity).view(B, -1))
        else:
            identity_proj = self.avg(identity).view(B, -1)
        
        # MLP projection (following original MambaNeck pattern)
        x_proj = self.mlp_proj(identity)  # [B, out_channels, H, W]
        x_proj = x_proj.permute(0, 2, 3, 1).view(B, H * W, -1)  # [B, H*W, out_channels]
        
        # Add shared positional embeddings (like original)
        x_proj = x_proj + self.pos_embed  # [B, H*W, out_channels]
        
        # Prepare spatial input for SS2D experts
        x_spatial = x_proj.view(B, H, W, -1)  # [B, H, W, out_channels]
        
        # SS2D MoE processing (each expert has its own SS2D block)
        moe_output, aux_loss = self.moe(x_spatial, training_phase=training_phase)
        
        # Initialize final output with MoE result
        final_output = moe_output
        
        # Collect skip connections for enhanced fusion
        skip_features = [identity_proj]
        
        # Multi-scale skip connections (Enhanced MoE feature)
        if self.use_multi_scale_skip:
            if multi_scale_features is not None:
                # Use actual multi-scale features when available
                for i, feat in enumerate(multi_scale_features):
                    if i < len(self.multi_scale_adapters):
                        # SS2D-based adapter processing (same as MambaNeck)
                        adapted_feat = self.multi_scale_adapters[i](feat)  # (B, out_channels)
                        skip_features.append(adapted_feat)
                        
                        # Log adapter usage for debugging
                        if hasattr(self, 'logger') and torch.rand(1).item() < 0.01:  # 1% 확률로 로그
                            self.logger.info(f"MoE SS2D Adapter {i}: {feat.shape} → {adapted_feat.shape}")

        # Cross-attention based skip connection fusion (Enhanced MoE)
        if self.use_multi_scale_skip and len(skip_features) > 1:
            # Stack all skip features: [B, num_features, feature_dim]
            skip_stack = torch.stack(skip_features, dim=1)  # [B, N, D]
            
            # Prepare Query (from MoE output), Key, Value (from skip features)
            query = self.query_proj(moe_output).unsqueeze(1)  # [B, 1, D]
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
            
            # 최종 출력: MoE + weighted skip connections
            final_output = moe_output + 0.1 * weighted_skip
            
            # 디버깅: cross-attention weights 출력 (MambaNeck과 동일한 형식)
            if hasattr(self, 'logger') and torch.rand(1).item() < 0.01:  # 1% 확률로 로그
                weight_values = weights[0].detach().cpu().numpy()
                # Generate dynamic feature names based on actual skip features (same as MambaNeck)
                feature_names = ['layer4(identity)']
                if self.use_multi_scale_skip:
                    feature_names.extend([f'layer{i+1}' for i in range(len(self.multi_scale_channels))])
                feature_names = feature_names[:len(skip_features)]
                weight_info = ', '.join([f"{name}: {val:.3f}" for name, val in zip(feature_names, weight_values)])
                self.logger.info(f"Cross-attention weights: {weight_info}")
        else:
            # Simple residual connection (when multi-scale is not used)
            final_output = moe_output + identity_proj
        
        # Prepare outputs
        outputs.update({
            'out': final_output,
            'aux_loss': aux_loss,
            'residual': identity_proj,
            'main': moe_output,  # For compatibility with existing code
        })
        
        # Add skip features for analysis (when multi-scale is enabled)
        if self.use_multi_scale_skip:
            outputs['skip_features'] = skip_features
        
        return outputs