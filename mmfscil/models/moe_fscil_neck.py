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
                 feat_size=7,
                 num_layers=2,
                 mid_channels=None):
        super(MultiScaleAdapter, self).__init__(init_cfg=None)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_size = feat_size
        self.num_layers = num_layers
        self.mid_channels = in_channels * 2 if mid_channels is None else mid_channels
        # 1. MoE와 동일한 크기로 맞춤
        self.spatial_adapter = nn.AdaptiveAvgPool2d((feat_size, feat_size))
        
        # 2. 간단한 MLP 프로젝션
        self.mlp_proj = self._build_mlp(in_channels, out_channels, self.mid_channels, num_layers, feat_size)
        
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
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Step 1: 공간 크기 통일
        x = self.spatial_adapter(x)  # (B, C, feat_size, feat_size)
        
        # Step 2: MLP 프로젝션
        x = self.mlp_proj(x)         # (B, out_channels, feat_size, feat_size)
        
        return x  # (B, out_channels, feat_size, feat_size) - 공간 정보 유지


class FSCILGate(nn.Module):
    def __init__(self,
                 dim,
                 num_experts: int,
                 top_k: int = 1,
                 capacity_factor: float = 1.25,
                 epsilon: float = 1e-6,
                 use_aux_loss: bool = True,
                 aux_loss_weight: float = 0.01):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.use_aux_loss = use_aux_loss
        self.aux_loss_weight = aux_loss_weight
        
        # Gating network
        self.w_gate = nn.Linear(dim, num_experts)

        
    def forward(self, x: torch.Tensor):

        # Compute raw gate scores directly (no session context for stability)
        raw_gate_scores = F.softmax(self.w_gate(x), dim=-1)  # [B, num_experts]
        
        # Determine capacity and apply top-k gating (configurable)
        capacity = int(self.capacity_factor * x.size(0))
        top_k_scores, top_k_indices = raw_gate_scores.topk(self.top_k, dim=-1)  # Use configurable top_k
        
        # Create sparsity mask
        mask = torch.zeros_like(raw_gate_scores).scatter_(1, top_k_indices, 1)
        masked_gate_scores = raw_gate_scores * mask
        
        # Normalize gate scores for dispatch
        denominators = masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        gate_scores = (masked_gate_scores / denominators) * capacity 
        
        # Compute auxiliary loss for load balancing (CORRECT Switch Transformer approach)
        aux_loss = None
        if self.use_aux_loss:
            # importance: 원본 softmax 확률의 평균 (soft routing 기준 기대 분포)
            importance = raw_gate_scores.mean(0)     # [num_experts]
            
            # load: 실제 top-1 dispatch 결과 (hard routing 분포)
            load = mask.float().mean(0)              # [num_experts]
            
            # load balancing loss: 진짜 "soft vs hard" 분포 비교
            # raw_gate_scores vs actual selection으로 제대로 된 load balancing
            #aux_loss = self.aux_loss_weight * ((load - importance) ** 2).mean()
            
            # Official Switch Transformer load balancing loss: importance * load (상관관계 최대화)
            # 공식: mean(importance * load) * num_experts²
            aux_loss = self.aux_loss_weight * (importance * load).mean() * (self.num_experts ** 2)
        
        return gate_scores, aux_loss
    



class SS2DExpert(nn.Module):
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

        self.norm = nn.LayerNorm(dim)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        B, H, W, dim = x.shape

        x_expert, _ = self.ss2d_block(x)  # [B, H, W, dim]

        x_expert = x_expert.permute(0, 3, 1, 2)  # [B, dim, H, W]
        x_expert = self.avg_pool(x_expert).view(B, -1)  # [B, dim]

        output = self.norm(x_expert)
        
        return output


class MoEFSCIL(nn.Module):

    def __init__(self,
                 dim,
                 num_experts=4,
                 top_k=2,
                 feat_size=7,
                 capacity_factor=1.25,
                 d_state=256,
                 dt_rank=256,
                 ssm_expand_ratio=1.0,
                 aux_loss_weight=0.01):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = FSCILGate(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            aux_loss_weight=aux_loss_weight
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
        
        # Flatten for gating decision (gate needs [B, dim] input)
        x_flat = F.adaptive_avg_pool2d(x.permute(0, 3, 1, 2), (1, 1)).view(B, -1)  # [B, dim]
        
        # Get expert selection probabilities and routing mask
        gate_scores, aux_loss = self.gate(x_flat)  # [B, num_experts]
        
        # Find top-k experts for each token (efficient sparse routing)
        top_k_scores, top_k_indices = gate_scores.topk(self.top_k, dim=-1)  # [B, top_k]
        
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
                 d_state=256,
                 dt_rank=None,
                 ssm_expand_ratio=1.0,
                 feat_size=2,
                 mid_channels=None,
                 num_layers=2,
                 loss_weight_supp=0.0,
                 loss_weight_supp_novel=0.0,
                 loss_weight_sep=0.0,
                 loss_weight_sep_new=0.0,
                 param_avg_dim='0-1-3',
                 use_multi_scale_skip=False,
                 multi_scale_channels=[128, 256, 512],
                 aux_loss_weight=0.01):
        super(MoEFSCILNeck, self).__init__(init_cfg=None)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.feat_size = feat_size
        self.mid_channels = in_channels * 2 if mid_channels is None else mid_channels
        self.num_layers = num_layers

        self.use_multi_scale_skip = use_multi_scale_skip
        self.multi_scale_channels = multi_scale_channels

        self.loss_weight_supp = loss_weight_supp
        self.loss_weight_supp_novel = loss_weight_supp_novel

        self.loss_weight_sep = loss_weight_sep
        self.loss_weight_sep_new = loss_weight_sep_new
        
        self.param_avg_dim = [int(item) for item in param_avg_dim.split('-')]
        
        self.logger = get_root_logger()
        self.logger.info(f"MoE-FSCIL Neck initialized: {num_experts} experts, top-{top_k} activation")
        
        if self.use_multi_scale_skip:
            self.logger.info(f"Enhanced MoE with Multi-Scale Skip Connections: {len(self.multi_scale_channels)} layers")
            self.logger.info(f"Multi-Scale Adapters: {self.multi_scale_channels} → {out_channels} channels")
            self.logger.info(f"Shared SS2D for skip connection processing after weighted combination")
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp_proj = self._build_mlp(
            in_channels, out_channels, self.mid_channels, num_layers, feat_size
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, feat_size * feat_size, out_channels)
        )
        trunc_normal_(self.pos_embed, std=.02)

        if self.use_multi_scale_skip:
            self.multi_scale_adapters = nn.ModuleList()
            for ch in self.multi_scale_channels:

                adapter = MultiScaleAdapter(
                    in_channels=ch,
                    out_channels=out_channels,
                    feat_size=self.feat_size,
                    num_layers=self.num_layers,
                    mid_channels=ch * 2 
                )
                self.multi_scale_adapters.append(adapter)
            
            directions = ('h', 'h_flip', 'v', 'v_flip')
            self.skip_ss2d = SS2D(
                out_channels,
                ssm_ratio=ssm_expand_ratio,
                d_state=d_state,
                dt_rank=dt_rank if dt_rank is not None else d_state,
                directions=directions,
                use_out_proj=False,
                use_out_norm=True
            )

        if self.use_multi_scale_skip:
            num_skip_sources = 1  # identity
            num_skip_sources += len(self.multi_scale_channels)
            
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
            self.query_proj = nn.Linear(out_channels, out_channels)
            self.key_proj = nn.Linear(out_channels, out_channels)
            self.value_proj = nn.Linear(out_channels, out_channels)

        self.moe = MoEFSCIL(
            dim=out_channels,
            num_experts=num_experts,
            top_k=top_k,
            feat_size=feat_size,
            d_state=d_state,
            dt_rank=dt_rank if dt_rank is not None else d_state,
            ssm_expand_ratio=ssm_expand_ratio,
            aux_loss_weight=aux_loss_weight
        )

        self.moe.debug_enabled = True
        self.moe.forward_count = 0 
        
        
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
        
        if self.use_multi_scale_skip:
            for i, adapter in enumerate(self.multi_scale_adapters):

                if hasattr(adapter, 'mlp_proj'):
                    first_layer = adapter.mlp_proj[0]  
                    if isinstance(first_layer, nn.Conv2d):
                        nn.init.kaiming_normal_(first_layer.weight, mode='fan_out', nonlinearity='relu')
                
                self.logger.info(f'Initialized MultiScaleAdapter {i} for channel {self.multi_scale_channels[i]} with {adapter.num_layers}-layer MLP')
            
            if hasattr(self, 'skip_ss2d'):
                with torch.no_grad():
                    if hasattr(self.skip_ss2d, 'in_proj'):
                        self.skip_ss2d.in_proj.weight.data *= 0.1
                self.logger.info('Initialized shared skip SS2D block for weighted feature processing')
    
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
        
        x_proj = self.mlp_proj(identity)  # [B, out_channels, H, W]
        x_proj = x_proj.permute(0, 2, 3, 1).view(B, H * W, -1)  # [B, H*W, out_channels]
        
        x_proj = x_proj + self.pos_embed  # [B, H*W, out_channels]
        
        x_spatial = x_proj.view(B, H, W, -1)  # [B, H, W, out_channels]

        moe_output, aux_loss = self.moe(x_spatial)

        final_output = moe_output

        skip_features_spatial = [identity] if self.use_multi_scale_skip else None  # 공간 정보 유지

        if self.use_multi_scale_skip and skip_features_spatial is not None:
            if multi_scale_features is not None:
                for i, feat in enumerate(multi_scale_features):
                    if i < len(self.multi_scale_adapters):
                        adapted_feat = self.multi_scale_adapters[i](feat)  # (B, out_channels, H, W)
                        skip_features_spatial.append(adapted_feat)

                        if hasattr(self, 'logger') and torch.rand(1).item() < 0.01:  # 1% 확률로 로그
                            self.logger.info(f"MultiScaleAdapter {i}: {feat.shape} → {adapted_feat.shape}")

        if self.use_multi_scale_skip and skip_features_spatial is not None and len(skip_features_spatial) > 1:
            # 모든 skip features는 이미 동일한 공간 크기 (MultiScaleAdapter에서 맞춤)
            B, C, H, W = skip_features_spatial[0].shape  # 기준 크기 (identity)
            
            skip_stack = torch.stack(skip_features_spatial, dim=1)  # [B, N, C, H, W]
            
            # 공간 차원을 유지하면서 가중치 계산을 위해 평균 풀링
            skip_flat = skip_stack.mean(dim=(-2, -1))  # [B, N, C] - 공간 정보 압축
            
            # Prepare Query (from MoE output), Key, Value (from skip features)
            query = self.query_proj(moe_output).unsqueeze(1)  # [B, 1, C]
            keys = self.key_proj(skip_flat)         # [B, N, C]
            values = self.value_proj(skip_flat)     # [B, N, C]
            
            # Multi-head cross-attention
            attended_features, attention_weights = self.cross_attention(query, keys, values)
            # attention_weights: [B, 1, N]
            
            # softmax 정규화 (안정성 보강)
            weights = torch.softmax(attention_weights.squeeze(1), dim=-1)  # [B, N]
            
            # Skip features에 가중치 적용 (공간 정보 유지)
            weighted_skip = (weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * skip_stack).sum(dim=1)  # [B, C, H, W]
            
            # Apply shared SS2D to weighted skip features (공간 정보 활용)
            weighted_skip_spatial = weighted_skip.permute(0, 2, 3, 1)  # [B, H, W, C]
            skip_ss2d_output, _ = self.skip_ss2d(weighted_skip_spatial)  # [B, H, W, C]
            
            # Convert back to vector format
            skip_ss2d_output = skip_ss2d_output.permute(0, 3, 1, 2)  # [B, C, H, W]
            skip_ss2d_output = F.adaptive_avg_pool2d(skip_ss2d_output, (1, 1)).view(B, -1)  # [B, C]
            
            # 최종 출력: MoE + SS2D-processed skip connections
            final_output = moe_output + 0.1 * skip_ss2d_output
            
            # 디버깅: cross-attention weights 출력 (MambaNeck과 동일한 형식)
            if hasattr(self, 'logger') and torch.rand(1).item() < 0.01:  # 1% 확률로 로그
                weight_values = weights[0].detach().cpu().numpy()
                # Generate dynamic feature names based on actual skip features (same as MambaNeck)
                feature_names = ['layer4']
                if self.use_multi_scale_skip:
                    feature_names.extend([f'layer{i+1}' for i in range(len(self.multi_scale_channels))])
                feature_names = feature_names[:len(skip_features_spatial)]
                weight_info = ', '.join([f"{name}: {val:.3f}" for name, val in zip(feature_names, weight_values)])
                self.logger.info(f"Cross-attention weights: {weight_info}")
        else:
            final_output = moe_output
        
        # Prepare outputs
        outputs.update({
            'out': final_output,
            'aux_loss': aux_loss,
            'main': moe_output,
        })

        if self.use_multi_scale_skip and skip_features_spatial is not None:
            outputs['skip_features'] = skip_features_spatial
        
        return outputs