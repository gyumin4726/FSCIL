import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from timm.models.layers import trunc_normal_

from mmcls.models.builder import NECKS
from mmcls.utils import get_root_logger

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
        
        # Simple linear projection for routing (official MoE style)
        self.gate = nn.Linear(dim, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        
    def forward(self, x: torch.Tensor):
        """
        Simplified linear projection based gating (official MoE style).
        
        Args:
            x: Input spatial features [B, H, W, dim]
            
        Returns:
            gate_scores: Expert selection scores [B, num_experts]
            aux_loss: Load balancing loss
        """
        B, H, W, dim = x.shape
        
        # Step 1: Spatial pooling (aggregate spatial information)
        x_pooled = x.mean(dim=[1, 2])  # [B, H, W, dim] -> [B, dim]
        
        # Step 2: Simple linear projection to expert logits
        gate_logits = self.gate(x_pooled)  # [B, dim] -> [B, num_experts]
        
        # Step 3: Softmax to get routing probabilities
        raw_gate_scores = F.softmax(gate_logits, dim=-1)  # [B, num_experts]
        
        # Step 4: Top-k selection
        current_top_k = self.top_k if self.training else self.eval_top_k
        top_k_scores, top_k_indices = raw_gate_scores.topk(current_top_k, dim=-1)  # [B, top_k]
        
        # Step 5: Create sparsity mask
        mask = torch.zeros_like(raw_gate_scores).scatter_(1, top_k_indices, 1)
        masked_gate_scores = raw_gate_scores * mask
        
        # Step 6: Keep original scores
        gate_scores = masked_gate_scores 
        
        # Step 7: Compute auxiliary loss for load balancing
        aux_loss = None
        if self.use_aux_loss:
            importance = raw_gate_scores.mean(0)     # [num_experts]
            load = (mask / current_top_k).float().mean(0)  # [num_experts]
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
                 eval_top_k=None,
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

        B, H, W, dim = x.shape
        
        # Pass spatial information directly to gate for spatial-aware routing
        gate_scores, aux_loss = self.gate(x)  # [B, num_experts] - x is [B, H, W, dim]
        
        # Find top-k experts for each token (학습/평가 모드에 따라 다른 top_k 사용)
        current_top_k = self.top_k if self.training else self.eval_top_k
        top_k_scores, top_k_indices = gate_scores.topk(current_top_k, dim=-1)  # [B, top_k]
        
        # 가중치 정규화: 선택된 experts의 가중치 합이 1이 되도록
        top_k_scores = F.softmax(top_k_scores, dim=-1)  # [B, top_k]
        
        # Debug: 10번째 forward마다 현재 배치, 100번째마다 누적 통계 출력
        if hasattr(self, 'debug_enabled') and self.debug_enabled:
            # Forward pass counter 증가
            if not hasattr(self, 'forward_count'):
                self.forward_count = 0
            self.forward_count += 1
            
            # 10번째 forward마다 현재 배치 통계 출력
            if self.forward_count % 10 == 0:
                # 현재 배치에서 각 expert 활성화 횟수 계산
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
                print(f"Forward #{self.forward_count:4d} | Batch Active: {active_count}/{self.num_experts} | {status_str}")
                print("=" * 100)
            
            # 100번째 forward마다 누적 통계 출력
            if self.forward_count % 469 == 0 and self.total_samples > 0:
                cumulative_counts = self.expert_activation_counts.cpu().numpy()
                total_samples = self.total_samples.item()
                cumulative_ratios = cumulative_counts / total_samples if total_samples > 0 else cumulative_counts
                
                cumulative_status = []
                for expert_id in range(self.num_experts):
                    ratio = cumulative_ratios[expert_id]
                    cumulative_status.append(f"{ratio*100:5.2f}%")
                
                cumulative_str = " | ".join([f"E{i}:{status}" for i, status in enumerate(cumulative_status)])
                print("🔥" * 50)
                print(f"📊 CUMULATIVE STATS (after {self.forward_count} forwards, {total_samples:,} samples)")
                print(f"{cumulative_str}")
                print("🔥" * 50)
        
        # Initialize output
        mixed_output = torch.zeros(B, dim, device=x.device, dtype=x.dtype)
        
        # Process only selected experts (sparse activation)
        for i in range(B):  # For each sample in batch
            for k in range(current_top_k):  # For each selected expert
                expert_idx = top_k_indices[i, k].item()
                expert_weight = top_k_scores[i, k]
                
                # SS2D expert processes spatial input
                expert_output = self.experts[expert_idx](x[i:i+1])  # [1, dim]
                mixed_output[i] += expert_weight * expert_output.squeeze(0)
                
                # 누적 통계 추적 (training 모드에서만)
                if self.training:
                    self.expert_activation_counts[expert_idx] += 1
        
        # 샘플 수 누적 (training 모드에서만)
        if self.training:
            self.total_samples += B
        
        return mixed_output, aux_loss


@NECKS.register_module()
class MoEFSCILNeckLinear(BaseModule):

    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 num_experts=4,
                 top_k=2,
                 eval_top_k=None,
                 feat_size=3,
                 use_aux_loss=True,
                 aux_loss_weight=0.01):
        super(MoEFSCILNeckLinear, self).__init__(init_cfg=None)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.eval_top_k = eval_top_k if eval_top_k is not None else top_k
        self.feat_size = feat_size
        self.use_aux_loss = use_aux_loss
        
        self.logger = get_root_logger()
        self.logger.info(f"MoE-FSCIL Neck (Linear) initialized: {num_experts} experts, top-{top_k} activation (train), top-{self.eval_top_k} activation (eval)")
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, feat_size * feat_size, out_channels)
        )
        trunc_normal_(self.pos_embed, std=.02)

        self.moe = MoEFSCIL(
            dim=out_channels,
            num_experts=num_experts,
            top_k=top_k,
            eval_top_k=self.eval_top_k,
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
        outputs = {
            'out': moe_output,
            'aux_loss': moe_aux_loss,
        }
        
        return outputs
