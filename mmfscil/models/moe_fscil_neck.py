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


class FSCILGate(nn.Module):
    def __init__(self,
                 dim,
                 num_experts: int,
                 top_k: int = 1,
                 eval_top_k: int = None,
                 use_aux_loss: bool = True,
                 aux_loss_weight: float = 0.01,
                 num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.eval_top_k = eval_top_k if eval_top_k is not None else top_k
        self.use_aux_loss = use_aux_loss
        self.aux_loss_weight = aux_loss_weight
        
        # Expert query embeddings for spatial-wise routing
        self.expert_queries = nn.Parameter(torch.randn(num_experts, dim))
        nn.init.xavier_uniform_(self.expert_queries)
        
        # Temperature for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

        
    def forward(self, x: torch.Tensor):
        """
        Spatial-wise routing with dot-product similarity
        - 각 spatial position마다 expert와의 유사도 계산
        - Pooling 없이 [B, H, W, num_experts] 출력
        
        Args:
            x: Input spatial features [B, H, W, dim]
            
        Returns:
            gate_scores: Spatial-wise expert scores [B, H, W, num_experts]
            aux_loss: Load balancing loss
        """
        B, H, W, dim = x.shape
        
        # Step 1: Compute similarity between each position and expert queries
        # x: [B, H, W, dim], expert_queries: [num_experts, dim]
        # Reshape for matrix multiplication
        x_flat = x.reshape(B * H * W, dim)  # [B*H*W, dim]
        
        # Dot product similarity (scaled)
        logits = x_flat @ self.expert_queries.T / self.temperature  # [B*H*W, num_experts]
        
        # Reshape back to spatial
        logits = logits.reshape(B, H, W, self.num_experts)  # [B, H, W, num_experts]
        
        # Step 2: Softmax over experts (각 position에서 expert 간 확률 분포)
        gate_scores = F.softmax(logits, dim=-1)  # [B, H, W, num_experts]
        
        # Step 3: Compute auxiliary loss for load balancing
        aux_loss = None
        if self.use_aux_loss:
            # Global average of gate scores across batch and spatial dimensions
            avg_gate_scores = gate_scores.mean(dim=[0, 1, 2])  # [num_experts]
            
            # Top-k selection for load calculation
            current_top_k = self.top_k if self.training else self.eval_top_k
            gate_scores_flat = gate_scores.view(-1, self.num_experts)  # [B*H*W, num_experts]
            _, top_k_indices = gate_scores_flat.topk(current_top_k, dim=-1)
            mask = torch.zeros_like(gate_scores_flat).scatter_(1, top_k_indices, 1)
            load = (mask / current_top_k).float().mean(0)  # [num_experts]
            
            # Load balancing loss
            aux_loss = self.aux_loss_weight * (avg_gate_scores * load).mean() * (self.num_experts ** 2)
        
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
        
    def forward(self, x):
        """
        Spatial-wise expert processing
        Args:
            x: [B, H, W, dim]
        Returns:
            output: [B, H, W, dim] - spatial features 유지
        """
        B, H, W, dim = x.shape

        x_expert, _ = self.ss2d_block(x)  # [B, H, W, dim]
        
        return x_expert


class MoEFSCIL(nn.Module):

    def __init__(self,
                 dim,
                 num_experts=4,
                 top_k=2,
                 eval_top_k=None,
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
        self.eval_top_k = eval_top_k if eval_top_k is not None else top_k

        self.gate = FSCILGate(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            eval_top_k=self.eval_top_k,
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
        
        # Expert 활성화 누적 통계 추적
        self.register_buffer('expert_activation_counts', torch.zeros(num_experts, dtype=torch.long))
        self.register_buffer('total_samples', torch.tensor(0, dtype=torch.long))
        
    def forward(self, x):
        """
        Spatial-wise MoE forward pass
        
        Args:
            x: [B, H, W, dim]
            
        Returns:
            output: [B, dim] - global pooled output
            aux_loss: Load balancing loss
        """
        B, H, W, dim = x.shape
        
        # Step 1: Get spatial-wise gate scores
        gate_scores, aux_loss = self.gate(x)  # [B, H, W, num_experts]
        
        # Step 2: Top-k selection per spatial position
        current_top_k = self.top_k if self.training else self.eval_top_k
        top_k_scores, top_k_indices = gate_scores.topk(current_top_k, dim=-1)  # [B, H, W, top_k]
        
        # 가중치 정규화: 선택된 experts의 가중치 합이 1이 되도록
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)  # [B, H, W, top_k]
        
        # Step 3: Compute all expert outputs (병렬 처리)
        expert_outputs = []
        for expert_idx in range(self.num_experts):
            expert_out = self.experts[expert_idx](x)  # [B, H, W, dim]
            expert_outputs.append(expert_out)
        expert_outputs = torch.stack(expert_outputs, dim=3)  # [B, H, W, num_experts, dim]
        
        # Step 4: Spatial-wise weighted sum
        # Create a mask for sparse routing
        mask = torch.zeros(B, H, W, self.num_experts, device=x.device)  # [B, H, W, num_experts]
        for k in range(current_top_k):
            expert_idx = top_k_indices[..., k]  # [B, H, W]
            weight = top_k_scores[..., k]  # [B, H, W]
            mask.scatter_(3, expert_idx.unsqueeze(-1), weight.unsqueeze(-1))
        
        # Apply mask and sum over experts
        mask = mask.unsqueeze(-1)  # [B, H, W, num_experts, 1]
        mixed_spatial = (expert_outputs * mask).sum(dim=3)  # [B, H, W, dim]
        
        # Step 5: Global pooling
        mixed_spatial = mixed_spatial.permute(0, 3, 1, 2)  # [B, dim, H, W]
        output = F.adaptive_avg_pool2d(mixed_spatial, (1, 1)).view(B, dim)  # [B, dim]
        
        # Debug: 10번째 forward마다 현재 배치 통계 출력
        if hasattr(self, 'debug_enabled') and self.debug_enabled:
            if not hasattr(self, 'forward_count'):
                self.forward_count = 0
            self.forward_count += 1
            
            # 10번째 forward마다 현재 배치 통계 출력
            if self.forward_count % 10 == 0:
                # Spatial-wise expert 활성화 횟수 계산
                expert_counts = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts)
                total_activations = expert_counts.sum().item()
                
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
                
                # Gate scores 평균 출력 (배치 및 spatial 평균)
                avg_gate_scores = gate_scores.mean(dim=[0, 1, 2])  # [num_experts]
                gate_scores_str = " | ".join([f"E{i}:{score.item()*100:5.2f}%" for i, score in enumerate(avg_gate_scores)])
                
                print("=" * 100)
                print(f"Forward #{self.forward_count:4d} | Spatial Active: {active_count}/{self.num_experts} | {status_str}")
                print(f"Gate Scores (spatial avg): {gate_scores_str}")
                print("=" * 100)
            
            # 100번째 forward마다 누적 통계 출력
            if self.forward_count % 94 == 0:
                # Spatial-wise routing에서는 누적 통계가 의미가 다름
                expert_counts = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts)
                total_activations = expert_counts.sum().item()
                
                cumulative_status = []
                for expert_id in range(self.num_experts):
                    count = expert_counts[expert_id].item()
                    ratio = count / total_activations if total_activations > 0 else 0.0
                    cumulative_status.append(f"{ratio*100:5.2f}%")
                
                cumulative_str = " | ".join([f"E{i}:{status}" for i, status in enumerate(cumulative_status)])
                print("🔥" * 50)
                print(f"📊 CUMULATIVE SPATIAL STATS (after {self.forward_count} forwards)")
                print(f"{cumulative_str}")
                print("🔥" * 50)
        
        return output, aux_loss


@NECKS.register_module()
class MoEFSCILNeck(BaseModule):

    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 num_experts=4,
                 top_k=2,
                 eval_top_k=None,
                 feat_size=3,
                 use_aux_loss=True,
                 aux_loss_weight=0.01):
        super(MoEFSCILNeck, self).__init__(init_cfg=None)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.eval_top_k = eval_top_k if eval_top_k is not None else top_k
        self.feat_size = feat_size
        self.use_aux_loss = use_aux_loss
        
        self.logger = get_root_logger()
        self.logger.info(f"MoE-FSCIL Neck initialized: {num_experts} experts, top-{top_k} activation (train), top-{self.eval_top_k} activation (eval)")
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.pos_embed = nn.Parameter(
            torch.zeros(1, feat_size * feat_size, out_channels)
        )
        trunc_normal_(self.pos_embed, std=.02)

        self.moe = MoEFSCIL(
            dim=out_channels,
            num_experts=num_experts,
            top_k=top_k,
            eval_top_k=self.eval_top_k,
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
    
    def forward(self, x):
        # Handle tuple input (e.g., from backbone with multiple outputs)
        if isinstance(x, tuple):
            x = x[-1]  # Use last layer (layer4) as main input

        B, C, H, W = x.shape
        outputs = {}
        
        # Prepare input for MoE
        x_proj = x.permute(0, 2, 3, 1).view(B, H * W, -1)  # [B, H*W, out_channels]
        x_proj = x_proj + self.pos_embed  # Add positional embedding
        x_spatial = x_proj.view(B, H, W, -1)  # [B, H, W, out_channels]

        # MoE processing
        moe_output, moe_aux_loss = self.moe(x_spatial)
        
        # Prepare outputs
        outputs.update({
            'out': moe_output,
            'aux_loss': moe_aux_loss,
            'main': moe_output,
        })
        
        return outputs
