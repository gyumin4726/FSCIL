import os
import sys
import torch
import torch.nn as nn
from mmcv import Config
from mmcls.models import build_classifier
import numpy as np
from typing import Dict, Any, Tuple

import mmfscil


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """모델의 파라미터 수를 계산합니다."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }




def try_thop_flops_component(component: nn.Module, input_shape: Tuple[int, ...], component_name: str) -> Tuple[int, int]:
    """컴포넌트별로 thop 라이브러리를 사용하여 FLOPs를 계산합니다."""
    try:
        from thop import profile
        
        # CUDA 사용 가능하면 CUDA로, 아니면 CPU로
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        component = component.to(device)
        component.eval()
        
        dummy_input = torch.randn(1, *input_shape).to(device)
        
        # ETFHead의 경우 특별 처리
        if 'ETFHead' in component_name:
            # ETFHead는 forward 대신 simple_test나 다른 메서드를 사용할 수 있음
            try:
                # 먼저 일반적인 forward 시도
                flops, params = profile(component, inputs=(dummy_input,), verbose=False)
            except:
                # forward가 없으면 파라미터만 계산
                params = sum(p.numel() for p in component.parameters())
                # ETFHead는 주로 Linear layer이므로 간단히 추정
                if hasattr(component, 'in_channels') and hasattr(component, 'num_classes'):
                    flops = component.in_channels * component.num_classes
                else:
                    flops = 0
                print(f"    💡 ETFHead는 표준 forward가 없어 추정값 사용")
                return int(flops), int(params)
        else:
            # thop을 사용하여 FLOPs와 파라미터 계산
            flops, params = profile(component, inputs=(dummy_input,), verbose=False)
        
        return int(flops), int(params)
    except ImportError:
        print("💡 thop 라이브러리가 설치되지 않았습니다. 설치하려면: pip install thop")
        return None, None
    except Exception as e:
        print(f"Warning: {component_name} thop FLOPs 계산 중 오류 발생: {e}")
        # 최소한 파라미터 수는 계산
        try:
            params = sum(p.numel() for p in component.parameters())
            return 0, int(params)  # FLOPs는 0으로, 파라미터는 실제 값
        except:
            return None, None


def analyze_moe_flops(neck: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, int]:
    """MoE Neck의 Train/Eval 모드별 FLOPs를 추정합니다."""
    flops_info = {
        'router': 0,
        'single_expert': 0,
        'train': 0,
        'eval': 0
    }
    
    if not hasattr(neck, 'moe'):
        return flops_info
    
    moe = neck.moe
    
    # Router FLOPs 추정 (Self-Attention + Cross-Attention + Projection)
    # 입력: [B, H*W, dim]
    B = 1
    H, W = 7, 7  # 일반적인 feature map 크기
    seq_len = H * W
    dim = getattr(moe, 'dim', 1024)
    num_heads = 8
    num_experts = getattr(moe, 'num_experts', 4)
    
    # Self-Attention FLOPs: Q, K, V projections + attention + output projection
    # QKV projection: 3 * (seq_len * dim * dim)
    # Attention: seq_len * seq_len * dim
    # Output projection: seq_len * dim * dim
    self_attn_flops = 3 * seq_len * dim * dim + seq_len * seq_len * dim + seq_len * dim * dim
    
    # Cross-Attention FLOPs (Query=features, Key&Value=expert_queries)
    # Q projection: seq_len * dim * dim
    # K, V projections: num_experts * dim * dim
    # Attention: seq_len * num_experts * dim
    # Output projection: seq_len * dim * dim
    cross_attn_flops = seq_len * dim * dim + 2 * num_experts * dim * dim + seq_len * num_experts * dim + seq_len * dim * dim
    
    # Gate projection: num_experts * dim
    gate_proj_flops = num_experts * dim
    
    flops_info['router'] = self_attn_flops + cross_attn_flops + gate_proj_flops
    
    # Single Expert FLOPs 추정 (SS2D는 매우 복잡함)
    if hasattr(moe, 'experts') and len(moe.experts) > 0:
        expert = moe.experts[0]
        if hasattr(expert, 'ss2d_block'):
            ss2d = expert.ss2d_block
            
            # SS2D 파라미터 추출
            d_model = dim
            ssm_ratio = getattr(ss2d, 'ssm_ratio', 2.0) if hasattr(ss2d, 'ssm_ratio') else 2.0
            d_expand = int(ssm_ratio * d_model)
            d_state = getattr(ss2d, 'd_state', 16) if hasattr(ss2d, 'd_state') else 16
            dt_rank = getattr(ss2d, 'dt_rank', d_model // 16) if hasattr(ss2d, 'dt_rank') else d_model // 16
            K = getattr(ss2d, 'K', 4) if hasattr(ss2d, 'K') else 4  # 방향 개수 (h, h_flip, v, v_flip)
            d_conv = getattr(ss2d, 'd_conv', 3) if hasattr(ss2d, 'd_conv') else 3
            
            seq_len = H * W
            
            # 1. Input projection: d_model → 2*d_expand
            in_proj_flops = seq_len * d_model * (2 * d_expand)
            
            # 2. Convolution (depthwise): d_expand channels, kernel_size=d_conv
            if d_conv > 1:
                conv_flops = seq_len * d_expand * (d_conv * d_conv)
            else:
                conv_flops = 0
            
            # 3. x_proj: d_inner → (dt_rank + d_state*2) for K directions
            d_inner = d_expand  # low rank인 경우 다를 수 있음
            x_proj_output_dim = dt_rank + d_state * 2
            x_proj_flops = K * seq_len * d_inner * x_proj_output_dim
            
            # 4. dt_proj: dt_rank → d_inner for K directions
            dt_proj_flops = K * seq_len * dt_rank * d_inner
            
            # 5. Selective Scan (가장 복잡한 부분!)
            # 각 방향마다 sequence를 따라 state update 수행
            # State update: d_inner * d_state * seq_len (per direction)
            # Total for K directions
            selective_scan_flops = K * d_inner * d_state * seq_len * 6  # 대략적 연산 복잡도
            
            # 6. Output projection (if used): d_expand → d_model
            out_proj_flops = seq_len * d_expand * d_model if getattr(ss2d, 'use_out_proj', True) else 0
            
            # 7. Layer Norm
            layer_norm_flops = seq_len * d_inner * 2
            
            # 8. Average Pooling (Expert의 마지막)
            avg_pool_flops = H * W * dim
            
            # 총합
            expert_flops = (in_proj_flops + conv_flops + x_proj_flops + dt_proj_flops + 
                          selective_scan_flops + out_proj_flops + layer_norm_flops + avg_pool_flops)
            
            flops_info['single_expert'] = expert_flops
            
            # 디버깅용 세부 정보 저장
            flops_info['expert_details'] = {
                'in_proj': in_proj_flops,
                'conv': conv_flops,
                'x_proj': x_proj_flops,
                'dt_proj': dt_proj_flops,
                'selective_scan': selective_scan_flops,
                'out_proj': out_proj_flops,
                'layer_norm': layer_norm_flops,
                'avg_pool': avg_pool_flops
            }
        else:
            # SS2D가 없는 경우 기본 추정
            expert_flops = dim * H * W * 100
            flops_info['single_expert'] = expert_flops
    
    # Train/Eval FLOPs
    train_top_k = getattr(moe, 'top_k', 2)
    eval_top_k = getattr(moe, 'eval_top_k', 1)
    
    flops_info['train'] = flops_info['router'] + flops_info['single_expert'] * train_top_k
    flops_info['eval'] = flops_info['router'] + flops_info['single_expert'] * eval_top_k
    
    return flops_info


def analyze_components_flops(model, input_shape: Tuple[int, ...] = (3, 224, 224)):
    """모델의 각 컴포넌트별로 FLOPs를 분석합니다."""
    print(f"\n⚡ 컴포넌트별 FLOPs 분석 (입력 크기: {input_shape})")
    print("=" * 60)
    
    total_flops = 0
    total_params = 0
    neck_train_flops = 0
    neck_eval_flops = 0
    
    # Backbone 분석 (FLOPs 계산은 생략, Neck 입력 크기만 확인)
    backbone_output_shape = (1024, 7, 7)  # 기본값
    
    if hasattr(model, 'backbone') and model.backbone is not None:
        print(f"\n🔧 Backbone: {type(model.backbone).__name__}")
        print("-" * 40)
        
        actual_params = sum(p.numel() for p in model.backbone.parameters())
        print(f"  🔢 파라미터:    {format_number(actual_params):>12}")
        print(f"  💡 FLOPs 계산은 생략합니다 (복잡한 구조)")
        total_params += actual_params
        
        # Backbone 출력 크기 자동 감지 (Neck 입력용)
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                dummy_input = torch.randn(1, *input_shape).to(device)
                model.backbone.eval()
                backbone_out = model.backbone(dummy_input)
                if isinstance(backbone_out, tuple):
                    backbone_out = backbone_out[-1]  # 마지막 feature map 사용
                _, C, H, W = backbone_out.shape
                backbone_output_shape = (C, H, W)
                print(f"  ✅ Neck 입력 크기: {backbone_output_shape}")
        except Exception as e:
            # 실패하면 neck의 in_channels 사용
            if hasattr(model, 'neck') and hasattr(model.neck, 'in_channels'):
                backbone_output_shape = (model.neck.in_channels, 7, 7)
                print(f"  💡 Neck 입력 크기 (추정): {backbone_output_shape}")
            else:
                print(f"  ⚠️ Neck 입력 크기 (기본값): {backbone_output_shape}")
    
    # Neck 분석 (MoE는 특별 처리)
    if hasattr(model, 'neck') and model.neck is not None:
        print(f"\n🔗 Neck: {type(model.neck).__name__}")
        print("-" * 40)
        try:
            actual_params = sum(p.numel() for p in model.neck.parameters())
            
            # MoE Neck인 경우 Train/Eval 별도 FLOPs 계산
            if 'MoE' in type(model.neck).__name__:
                print(f"  💡 MoE Neck은 Train/Eval 모드별로 FLOPs가 다릅니다")
                moe_flops = analyze_moe_flops(model.neck, backbone_output_shape)
                
                print(f"  🔢 파라미터:         {format_number(actual_params):>12}")
                print(f"\n  📊 Router FLOPs:     {format_number(moe_flops['router']):>12}")
                print(f"  📊 Single Expert:    {format_number(moe_flops['single_expert']):>12}")
                
                # Expert 세부 FLOPs 출력 (있는 경우)
                if 'expert_details' in moe_flops:
                    details = moe_flops['expert_details']
                    print(f"     ├─ Input Proj:       {format_number(details['in_proj']):>10}")
                    print(f"     ├─ Conv2D:           {format_number(details['conv']):>10}")
                    print(f"     ├─ X Projection:     {format_number(details['x_proj']):>10}")
                    print(f"     ├─ DT Projection:    {format_number(details['dt_proj']):>10}")
                    print(f"     ├─ Selective Scan:   {format_number(details['selective_scan']):>10} 👈 핵심!")
                    print(f"     ├─ Output Proj:      {format_number(details['out_proj']):>10}")
                    print(f"     ├─ Layer Norm:       {format_number(details['layer_norm']):>10}")
                    print(f"     └─ Avg Pool:         {format_number(details['avg_pool']):>10}")
                
                print(f"\n  ⚡ Train FLOPs:      {format_number(moe_flops['train']):>12} (top-k experts)")
                print(f"  ⚡ Eval FLOPs:       {format_number(moe_flops['eval']):>12} (top-k experts)")
                
                neck_train_flops = moe_flops['train']
                neck_eval_flops = moe_flops['eval']
                total_params += actual_params
            else:
                # 일반 Neck
                neck_input_shape = backbone_output_shape
                neck_flops, thop_params = try_thop_flops_component(model.neck, neck_input_shape, "Neck")
                
                if neck_flops is not None:
                    print(f"  📊 FLOPs:      {format_number(neck_flops):>12}")
                    print(f"  🔢 파라미터:    {format_number(actual_params):>12}")
                    total_flops += neck_flops
                    total_params += actual_params
                    neck_train_flops = neck_eval_flops = neck_flops
                else:
                    print("  ❌ FLOPs 계산 실패")
                    print(f"  🔢 파라미터:    {format_number(actual_params):>12}")
                    total_params += actual_params
        except Exception as e:
            print(f"  ❌ FLOPs 계산 실패: {e}")
    
    # Head 분석 (ETFHead는 파라미터가 없으므로 생략)
    # ETFHead는 고정된 ETF classifier로 학습 가능한 파라미터가 없음

    if neck_train_flops > 0:
        print(f"\n🏆 Neck FLOPs 요약")
        print("=" * 60)
        
        print(f"  📊 Train FLOPs (Neck): {format_number(neck_train_flops):>12}")
        print(f"  📊 Eval FLOPs (Neck):  {format_number(neck_eval_flops):>12}")
        
        flops_reduction = neck_train_flops - neck_eval_flops
        reduction_ratio = (flops_reduction / neck_train_flops * 100) if neck_train_flops > 0 else 0
        print(f"\n  💡 Eval FLOPs 절감:   {format_number(flops_reduction):>12} ({reduction_ratio:.1f}% 감소)")
        print(f"  🔢 총 파라미터:        {format_number(total_params):>12}")
    
    return total_flops, total_params


def format_number(num: int) -> str:
    """숫자를 읽기 쉬운 형태로 포맷합니다."""
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def analyze_model_components(model: nn.Module) -> Dict[str, Dict[str, int]]:
    """모델의 각 컴포넌트별 파라미터 수를 분석합니다."""
    component_stats = {}
    
    for name, module in model.named_children():
        params = count_parameters(module)
        component_stats[name] = params
    
    return component_stats


def analyze_moe_neck(neck: nn.Module) -> Dict[str, Any]:
    """MoE Neck의 세부 구조를 분석합니다."""
    stats = {
        'total_params': sum(p.numel() for p in neck.parameters()),
        'components': {}
    }
    
    # MoE 모듈 분석
    if hasattr(neck, 'moe'):
        moe = neck.moe
        
        # Gate (Router) 분석
        if hasattr(moe, 'gate'):
            gate_params = sum(p.numel() for p in moe.gate.parameters())
            stats['components']['gate_router'] = gate_params
            
            # Gate 내부 세부 분석
            gate_details = {}
            if hasattr(moe.gate, 'spatial_self_attention'):
                gate_details['self_attention'] = sum(p.numel() for p in moe.gate.spatial_self_attention.parameters())
            if hasattr(moe.gate, 'expert_cross_attention'):
                gate_details['cross_attention'] = sum(p.numel() for p in moe.gate.expert_cross_attention.parameters())
            if hasattr(moe.gate, 'expert_queries'):
                gate_details['expert_queries'] = moe.gate.expert_queries.numel()
            if hasattr(moe.gate, 'gate_proj'):
                gate_details['gate_proj'] = sum(p.numel() for p in moe.gate.gate_proj.parameters())
            
            stats['components']['gate_details'] = gate_details
        
        # Experts 분석
        if hasattr(moe, 'experts'):
            experts = moe.experts
            num_experts = len(experts)
            single_expert_params = sum(p.numel() for p in experts[0].parameters()) if num_experts > 0 else 0
            total_experts_params = sum(p.numel() for p in experts.parameters())
            
            stats['components']['experts'] = {
                'num_experts': num_experts,
                'single_expert': single_expert_params,
                'total_experts': total_experts_params
            }
            
            # Top-K 정보
            if hasattr(moe, 'top_k') and hasattr(moe, 'eval_top_k'):
                stats['top_k_info'] = {
                    'train_top_k': moe.top_k,
                    'eval_top_k': moe.eval_top_k,
                    'train_activation_ratio': moe.top_k / num_experts,
                    'eval_activation_ratio': moe.eval_top_k / num_experts
                }
    
    # Position embedding
    if hasattr(neck, 'pos_embed'):
        stats['components']['pos_embed'] = neck.pos_embed.numel()
    
    return stats


def print_model_analysis(model: nn.Module, config_name: str, input_shape: Tuple[int, ...] = (3, 224, 224)):
    """모델 분석 결과를 출력합니다."""
    print("=" * 80)
    print(f"🔍 CUB 모델 분석 결과 - {config_name}")
    print("=" * 80)
    
    # 파라미터 수 계산
    param_stats = count_parameters(model)
    print(f"\n📊 파라미터 통계:")
    print("-" * 50)
    print(f"  전체 파라미터:     {format_number(param_stats['total']):>12} ({param_stats['total']:,})")
    print(f"  훈련 가능:        {format_number(param_stats['trainable']):>12} ({param_stats['trainable']:,})")
    print(f"  고정됨:          {format_number(param_stats['frozen']):>12} ({param_stats['frozen']:,})")
    
    # 컴포넌트별 분석
    print(f"\n🧩 컴포넌트별 파라미터:")
    print("-" * 50)
    component_stats = analyze_model_components(model)
    for comp_name, stats in component_stats.items():
        print(f"  {comp_name:12}: {format_number(stats['total']):>12} ({stats['total']:,})")
    
    # MoE Neck 세부 분석
    if hasattr(model, 'neck'):
        moe_stats = analyze_moe_neck(model.neck)
        
        print(f"\n🔀 MoE Neck 세부 분석:")
        print("-" * 50)
        print(f"  총 Neck 파라미터: {format_number(moe_stats['total_params']):>12}")
        
        if 'gate_router' in moe_stats['components']:
            gate_params = moe_stats['components']['gate_router']
            gate_ratio = gate_params / moe_stats['total_params'] * 100
            print(f"\n  🎯 Router (Gate):  {format_number(gate_params):>12} ({gate_ratio:.1f}% of Neck)")
            
            # Gate 세부 구조
            if 'gate_details' in moe_stats['components']:
                details = moe_stats['components']['gate_details']
                print(f"     ├─ Self-Attention:   {format_number(details.get('self_attention', 0)):>10}")
                print(f"     ├─ Cross-Attention:  {format_number(details.get('cross_attention', 0)):>10}")
                print(f"     ├─ Expert Queries:   {format_number(details.get('expert_queries', 0)):>10}")
                print(f"     └─ Gate Projection:  {format_number(details.get('gate_proj', 0)):>10}")
        
        if 'experts' in moe_stats['components']:
            exp_info = moe_stats['components']['experts']
            experts_ratio = exp_info['total_experts'] / moe_stats['total_params'] * 100
            print(f"\n  🤖 Experts ({exp_info['num_experts']}개):   {format_number(exp_info['total_experts']):>12} ({experts_ratio:.1f}% of Neck)")
            print(f"     └─ 단일 Expert:     {format_number(exp_info['single_expert']):>10}")
        
        if 'pos_embed' in moe_stats['components']:
            pos_params = moe_stats['components']['pos_embed']
            pos_ratio = pos_params / moe_stats['total_params'] * 100
            print(f"\n  📍 Position Embed: {format_number(pos_params):>12} ({pos_ratio:.1f}% of Neck)")
        
        # Top-K 활성화 정보
        if 'top_k_info' in moe_stats:
            info = moe_stats['top_k_info']
            print(f"\n  ⚡ 실제 활성화 파라미터 (Expert만 계산):")
            print(f"     ├─ Train (top-{info['train_top_k']}): {format_number(exp_info['single_expert'] * info['train_top_k']):>10} ({info['train_activation_ratio']*100:.0f}% experts)")
            print(f"     └─ Eval  (top-{info['eval_top_k']}): {format_number(exp_info['single_expert'] * info['eval_top_k']):>10} ({info['eval_activation_ratio']*100:.0f}% experts)")
            
            # 전체 모델 기준 실제 활성화 파라미터
            total_params = param_stats['total']
            router_and_pos = gate_params + pos_params
            active_train = router_and_pos + (exp_info['single_expert'] * info['train_top_k'])
            active_eval = router_and_pos + (exp_info['single_expert'] * info['eval_top_k'])
            
            backbone_params = component_stats.get('backbone', {}).get('total', 0)
            head_params = component_stats.get('head', {}).get('total', 0)
            
            print(f"\n  💡 전체 모델 기준 활성화 파라미터:")
            print(f"     ├─ Backbone:        {format_number(backbone_params):>10}")
            print(f"     ├─ Neck (활성화):   {format_number(active_train):>10} (Train) / {format_number(active_eval):>10} (Eval)")
            print(f"     ├─ Head:            {format_number(head_params):>10}")
            print(f"     ├─ Train 합계:      {format_number(backbone_params + active_train + head_params):>10}")
            print(f"     └─ Eval  합계:      {format_number(backbone_params + active_eval + head_params):>10}")
    
    # 컴포넌트별 FLOPs 분석
    analyze_components_flops(model, input_shape)


def main():
    print("🚀 CUB 모델 분석을 시작합니다...")
    print("=" * 80)
    
    # CUB 설정 파일들
    configs = {
        "CUB Base": "configs/cub/cub_base.py",
        "CUB Incremental": "configs/cub/cub_inc.py"
    }
    
    # CUB 이미지 크기 (일반적으로 224x224)
    input_shape = (3, 224, 224)
    
    for config_name, config_path in configs.items():
        if not os.path.exists(config_path):
            print(f"\n❌ 설정 파일을 찾을 수 없습니다: {config_path}")
            continue
            
        try:
            print(f"\n🔄 {config_name} 모델 로딩 중...")
            print("-" * 40)
            
            # 설정 파일 로드
            cfg = Config.fromfile(config_path)
            
            # 모델 빌드
            model = build_classifier(cfg.model)
            
            # 모델 분석 실행
            print_model_analysis(model, config_name, input_shape)
            
        except Exception as e:
            print(f"\n❌ {config_name} 모델 분석 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("✅ 모델 분석이 완료되었습니다!")

if __name__ == '__main__':
    main()
