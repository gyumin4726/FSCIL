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


class MultiScaleAdapter(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels=512,
                 feat_size=7):
        super(MultiScaleAdapter, self).__init__(init_cfg=None)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_size = feat_size

        # 1. MoE와 동일한 크기로 맞춤
        self.spatial_adapter = nn.AdaptiveAvgPool2d((feat_size, feat_size))
        
        # 2. 간단한 MLP 프로젝션
        # MLP 제거 - 단순한 1x1 conv로 채널 수만 맞춤
        self.mlp_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Step 1: 공간 크기 통일
        x = self.spatial_adapter(x)  # (B, C, feat_size, feat_size)
        
        # Step 2: MLP 프로젝션
        x = self.mlp_proj(x)         # (B, out_channels, feat_size, feat_size)
        
        return x  # (B, out_channels, feat_size, feat_size) - 공간 정보 유지


class MultiScaleRouter(nn.Module):
    """Multi-Scale Layer Router for dynamic auxiliary layer weighting"""
    def __init__(self,
                 dim,
                 num_aux_layers: int = 3,  # layer1, layer2, layer3
                 num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_aux_layers = num_aux_layers
        self.num_heads = num_heads
        
        # Self-attention for spatial context learning
        self.spatial_self_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Cross-attention for auxiliary layer routing
        self.aux_layer_cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, layer4_features: torch.Tensor, aux_layer_features: list):
        """
        Multi-Scale Auxiliary Layer Router
        
        Args:
            layer4_features: Layer4 spatial features [B, H, W, dim] (base/query)
            aux_layer_features: List of auxiliary layer features [layer1, layer2, layer3]
            
        Returns:
            aux_weights: Auxiliary layer weights [B, num_aux_layers] (can be > 1.0)
        """
        B, H, W, dim = layer4_features.shape
        
        # Step 1: Convert layer4 (query) to sequence format
        layer4_spatial = layer4_features.view(B, H * W, dim)  # [B, H*W, dim]
        
        # Step 2: Self-attention on layer4 for spatial context learning
        contextualized_layer4, _ = self.spatial_self_attention(
            query=layer4_spatial,
            key=layer4_spatial,
            value=layer4_spatial
        )
        
        # Step 3: Prepare auxiliary layer features as key/value
        # Stack auxiliary layers and convert to sequence format
        aux_stack = torch.stack(aux_layer_features, dim=1)  # [B, num_aux_layers, C, H, W]
        aux_spatial = aux_stack.view(B, self.num_aux_layers, H * W, dim)  # [B, num_aux_layers, H*W, dim]
        
        # Global average pooling for each auxiliary layer to get layer-level representations
        aux_layer_reps = aux_spatial.mean(dim=2)  # [B, num_aux_layers, dim]
        
        # Step 4: Cross-attention between layer4 (query) and auxiliary layers (key/value)
        attended_features, attention_weights = self.aux_layer_cross_attention(
            query=contextualized_layer4,  # [B, H*W, dim] - layer4 contextualized features
            key=aux_layer_reps,          # [B, num_aux_layers, dim] - auxiliary layer representations
            value=aux_layer_reps         # [B, num_aux_layers, dim] - auxiliary layer representations
        )
        
        # Step 5: Aggregate spatial attention to get auxiliary layer importance
        spatial_aux_scores = attention_weights.mean(dim=1)  # [B, num_aux_layers]
        
        # Step 6: Generate auxiliary weights (NO softmax - can be any positive value)
        aux_weights = torch.sigmoid(spatial_aux_scores)  # [B, num_aux_layers] - range [0, 1]
        
        return aux_weights


class FSCILGate(nn.Module):
    def __init__(self,
                 dim,
                 num_experts: int,
                 top_k: int = 1,
                 use_aux_loss: bool = True,
                 aux_loss_weight: float = 0.01,
                 num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss
        self.aux_loss_weight = aux_loss_weight
        self.num_heads = num_heads
        
        # Self-attention for spatial context learning (Query 생성용)
        self.spatial_self_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Cross-attention for expert routing
        self.expert_cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Expert query embeddings for spatial routing
        self.expert_queries = nn.Parameter(torch.randn(num_experts, dim))
        nn.init.xavier_uniform_(self.expert_queries)
        
        # Spatial gating projection
        self.gate_proj = nn.Linear(dim, num_experts)

        
    def forward(self, x: torch.Tensor):
        """
        Self-attention + Cross-attention based gating
        - Self-attention으로 spatial context 학습
        - Cross-attention으로 expert routing
        
        Args:
            x: Input spatial features [B, H, W, dim]
            
        Returns:
            gate_scores: Expert selection scores [B, num_experts]
            aux_loss: Load balancing loss
        """
        B, H, W, dim = x.shape
        
        # Step 1: Convert to sequence format
        x_spatial = x.view(B, H * W, dim)  # [B, H*W, dim]
        
        # Step 2: Self-attention for spatial context learning (Query 생성)
        contextualized_features, _ = self.spatial_self_attention(
            query=x_spatial,    # [B, H*W, dim]
            key=x_spatial,      # [B, H*W, dim] 
            value=x_spatial     # [B, H*W, dim]
        )
        
        # Step 3: Expert queries 준비
        expert_queries = self.expert_queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_experts, dim]
        
        # Step 4: Cross-attention for expert routing
        # Query: contextualized features, Key&Value: expert queries
        attended_features, attention_weights = self.expert_cross_attention(
            query=contextualized_features,  # [B, H*W, dim] - self-attention으로 contextualized된 features
            key=expert_queries,            # [B, num_experts, dim]
            value=expert_queries           # [B, num_experts, dim]
        )
        
        # Step 5: 공간별 expert 선호도를 global expert 선호도로 집계
        # attention_weights: [B, H*W, num_experts] - 각 spatial position의 expert 선호도
        spatial_expert_scores = attention_weights.mean(dim=1)  # [B, num_experts] - 공간 평균
        
        # Step 6: 최종 gate scores 생성
        # Cross-attention 결과만 사용
        raw_gate_scores = F.softmax(spatial_expert_scores, dim=-1)  # [B, num_experts]
        
        # Step 7: Top-k selection (no capacity constraints - batch-level routing)
        top_k_scores, top_k_indices = raw_gate_scores.topk(self.top_k, dim=-1)  # [B, top_k]
        
        # Create sparsity mask
        mask = torch.zeros_like(raw_gate_scores).scatter_(1, top_k_indices, 1)
        masked_gate_scores = raw_gate_scores * mask
        
        # Normalize gate scores
        gate_scores = masked_gate_scores 
        
        # Step 8: Compute auxiliary loss for load balancing
        aux_loss = None
        if self.use_aux_loss:
            # importance: 원본 softmax 확률의 평균 (soft routing 기준 기대 분포)
            importance = raw_gate_scores.mean(0)     # [num_experts]
            
            # load: 실제 top-k dispatch 결과 (hard routing 분포)
            # Normalize by top_k to maintain consistent loss scale
            load = (mask / self.top_k).float().mean(0)  # [num_experts]
            
            # Official Switch Transformer load balancing loss
            aux_loss = self.aux_loss_weight * (importance * load).mean() * (self.num_experts ** 2)
        
        return gate_scores, aux_loss
    



class SS2DExpert(nn.Module):
    def __init__(self,
                 dim,
                 expert_id=0,
                 d_state=1,
                 dt_rank=4,
                 ssm_expand_ratio=1.0):
        super().__init__()
        self.dim = dim
        self.expert_id = expert_id
        self.d_state = d_state
        self.dt_rank = dt_rank
        
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

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        B, H, W, dim = x.shape

        x_expert, _ = self.ss2d_block(x)  # [B, H, W, dim]

        x_expert = x_expert.permute(0, 3, 1, 2)  # [B, dim, H, W]
        x_expert = self.avg_pool(x_expert).view(B, -1)  # [B, dim]

        output = x_expert
        
        return output


class MoEFSCIL(nn.Module):

    def __init__(self,
                 dim,
                 num_experts=4,
                 top_k=2,
                 feat_size=7,
                 d_state=1,
                 dt_rank=4,
                 ssm_expand_ratio=1.0,
                 use_aux_loss=True,
                 aux_loss_weight=0.01,
                 num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = FSCILGate(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            use_aux_loss=use_aux_loss,
            aux_loss_weight=aux_loss_weight,
            num_heads=num_heads
        )
        
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
        
    def forward(self, x):

        B, H, W, dim = x.shape
        
        # Pass spatial information directly to gate for spatial-aware routing
        gate_scores, aux_loss = self.gate(x)  # [B, num_experts] - x is [B, H, W, dim]
        
        # Find top-k experts for each token (efficient sparse routing)
        top_k_scores, top_k_indices = gate_scores.topk(self.top_k, dim=-1)  # [B, top_k]
        
        # 가중치 정규화: 선택된 experts의 가중치 합이 1이 되도록
        top_k_scores = F.softmax(top_k_scores, dim=-1)  # [B, top_k]
        
        # Debug: 10번째 forward마다 출력)
        if hasattr(self, 'debug_enabled') and self.debug_enabled:
            # Forward pass counter 증가
            if not hasattr(self, 'forward_count'):
                self.forward_count = 0
            self.forward_count += 1
            
            # 10번째 forward마다 출력
            if self.forward_count % 10 == 0:
                # 전체 배치에서 각 expert 활성화 횟수 계산
                expert_counts = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts)
                total_activations = expert_counts.sum().item()
                
                # 모든 experts 상태를 표시 (활성화되지 않은 것은 -)
                expert_status = []
                active_count = 0
                for expert_id in range(self.num_experts):
                    count = expert_counts[expert_id].item()
                    if count > 0:
                        ratio = count / total_activations if total_activations > 0 else 0.0
                        expert_status.append(f"{ratio*100:4.1f}%")
                        active_count += 1
                    else:
                        expert_status.append("  - ")

                status_str = " | ".join([f"E{i}:{status}" for i, status in enumerate(expert_status)])
                print("=" * 100)
                print(f"Forward #{self.forward_count:4d} | Active: {active_count}/{self.num_experts} | {status_str}")
                print("=" * 100)
        
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

    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 num_experts=4,
                 top_k=2,
                 feat_size=3,
                 use_multi_scale_skip=False,
                 multi_scale_channels=[128, 256, 512],
                 use_aux_loss=True,
                 aux_loss_weight=0.01):
        super(MoEFSCILNeck, self).__init__(init_cfg=None)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.feat_size = feat_size

        self.use_multi_scale_skip = use_multi_scale_skip
        self.multi_scale_channels = multi_scale_channels
        self.use_aux_loss = use_aux_loss

        
        self.logger = get_root_logger()
        self.logger.info(f"MoE-FSCIL Neck initialized: {num_experts} experts, top-{top_k} activation")
        
        if self.use_multi_scale_skip:
            self.logger.info(f"Enhanced MoE with Layer4-based Auxiliary Fusion: {len(self.multi_scale_channels)} auxiliary layers")
            self.logger.info(f"Multi-Scale Adapters: {self.multi_scale_channels} → {out_channels} channels")
            self.logger.info(f"Dynamic Auxiliary Router with {num_heads}-head attention for auxiliary layer weighting")
            self.logger.info(f"Layer4 (base) + dynamically weighted auxiliary layers (layer1,2,3) → MoE")
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.pos_embed = nn.Parameter(
            torch.zeros(1, feat_size * feat_size, out_channels)
        )
        trunc_normal_(self.pos_embed, std=.02)

        if self.use_multi_scale_skip:
            self.multi_scale_adapters = nn.ModuleList()
            # layer4는 이미 out_channels와 동일하므로 adapter 불필요
            # layer1, layer2, layer3만 adapter 필요
            for ch in self.multi_scale_channels:
                adapter = MultiScaleAdapter(
                    in_channels=ch,
                    out_channels=out_channels,
                    feat_size=self.feat_size,
                )
                self.multi_scale_adapters.append(adapter)
            
            # Multi-Scale Auxiliary Router 추가
            self.multi_scale_router = MultiScaleRouter(
                dim=out_channels,
                num_aux_layers=len(self.multi_scale_channels),  # [layer1, layer2, layer3] only
                num_heads=8
            )

        self.moe = MoEFSCIL(
            dim=out_channels,
            num_experts=num_experts,
            top_k=top_k,
            feat_size=feat_size,
            d_state=1,
            dt_rank=4,
            ssm_expand_ratio=1.0,
            use_aux_loss=use_aux_loss,
            aux_loss_weight=aux_loss_weight,
            num_heads=8
        )

        self.moe.debug_enabled = True
        self.moe.forward_count = 0 
        
        # 초기화 플래그 추가
        self._weights_initialized = False
        
    def init_weights(self):
        """Initialize weights with proper scaling for MoE and multi-scale adapters."""
        # 이미 초기화되었으면 스킵
        if self._weights_initialized:
            return
            
        self.logger.info("🔧 Initializing MoE-FSCIL Neck weights...")
        
        if self.use_multi_scale_skip:
            for i, adapter in enumerate(self.multi_scale_adapters):
                if hasattr(adapter, 'mlp_proj'):
                    # mlp_proj는 단일 Conv2d 레이어이므로 직접 접근
                    if isinstance(adapter.mlp_proj, nn.Conv2d):
                        nn.init.kaiming_normal_(adapter.mlp_proj.weight, mode='fan_out', nonlinearity='relu')
                
                self.logger.info(f'Initialized MultiScaleAdapter {i} for channel {self.multi_scale_channels[i]} with 1x1 conv')
            
            # Multi-Scale Auxiliary Router 초기화
            if hasattr(self, 'multi_scale_router'):
                self.logger.info('Initialized Multi-Scale Auxiliary Router for dynamic auxiliary layer weighting')
        
        # 초기화 완료 플래그 설정
        self._weights_initialized = True
    
    def forward(self, x, multi_scale_features=None):

        if isinstance(x, tuple):
            x = x[-1]  # layer4 as main input
            if self.use_multi_scale_skip and multi_scale_features is None and len(x) > 1:
                multi_scale_features = x[:-1]  # [layer1, layer2, layer3]
        
        # multi_scale_features가 없으면 오류 발생 (multi-scale 사용시에만)
        if self.use_multi_scale_skip and (multi_scale_features is None or len(multi_scale_features) == 0):
            raise ValueError('use_multi_scale_skip=True 인데 multi_scale_features가 없습니다. backbone에서 여러 레이어 출력을 반환하도록 설정하세요.')

        B, C, H, W = x.shape
        identity = x
        outputs = {}
        
        # ===== NEW: Layer4-based Multi-Scale Auxiliary Fusion =====
        if self.use_multi_scale_skip:
            # Step 1: Prepare auxiliary layer features (layer1, layer2, layer3)
            aux_layer_features = []
            if multi_scale_features is not None:
                for i, feat in enumerate(multi_scale_features):
                    if i < len(self.multi_scale_adapters):
                        adapted_feat = self.multi_scale_adapters[i](feat)  # (B, out_channels, feat_size, feat_size)
                        aux_layer_features.append(adapted_feat)
            
            # Step 2: Get auxiliary weights from Multi-Scale Router
            # Query: layer4, Key/Value: auxiliary layers (layer1, layer2, layer3)
            layer4_spatial = identity.permute(0, 2, 3, 1)  # [B, H, W, out_channels]
            aux_weights = self.multi_scale_router(layer4_spatial, aux_layer_features)  # [B, num_aux_layers]
            
            # Step 3: Apply auxiliary weights to auxiliary layers
            auxiliary_contribution = torch.zeros_like(identity)  # [B, C, H, W]
            for i, aux_feat in enumerate(aux_layer_features):
                auxiliary_contribution += aux_weights[:, i:i+1, None, None] * aux_feat
            
            # Step 4: Final fusion = Layer4 (base) + weighted auxiliary contributions
            x_for_moe = identity + auxiliary_contribution  # Layer4는 그대로, 보조 레이어들만 가중치로 추가
            
            # Debug logging
            if hasattr(self, 'logger') and torch.rand(1).item() < 0.01:  # 1% 확률로 로그
                weight_values = aux_weights[0].detach().cpu().numpy()
                aux_names = [f'layer{i+1}' for i in range(len(self.multi_scale_channels))]
                weight_info = ', '.join([f"{name}: {val:.3f}" for name, val in zip(aux_names, weight_values)])
                self.logger.info(f"Auxiliary Layer Weights (added to Layer4): {weight_info}")
        else:
            # No multi-scale fusion, use original layer4
            x_for_moe = identity
        
        # Step 5: Prepare input for MoE
        x_proj = x_for_moe  # Use fused features instead of just layer4
        x_proj = x_proj.permute(0, 2, 3, 1).view(B, H * W, -1)  # [B, H*W, out_channels]
        
        x_proj = x_proj + self.pos_embed  # [B, H*W, out_channels]
        
        x_spatial = x_proj.view(B, H, W, -1)  # [B, H, W, out_channels]

        # Step 6: MoE processing with fused multi-scale features
        moe_output, moe_aux_loss = self.moe(x_spatial)

        # ===== REMOVED: Old residual skip connection logic =====
        # Multi-scale fusion is now done BEFORE MoE, so no need for post-MoE residual
        final_output = moe_output
        
        # Prepare outputs
        outputs.update({
            'out': final_output,
            'aux_loss': moe_aux_loss,
            'main': moe_output,
        })
        
        if self.use_multi_scale_skip:
            if 'all_layer_features' in locals():
                outputs['fused_features'] = x_for_moe
        
        return outputs
