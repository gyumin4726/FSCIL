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


def estimate_neck_flops_from_structure(
    neck: nn.Module, 
    seq_len: int, 
    train_top_k: int = None, 
    eval_top_k: int = None,
    num_experts: int = None
) -> Dict[str, int]:
    """Neck 구조를 분석하여 FLOPs를 추정합니다."""
    
    result = {
        'neck_train': 0,
        'neck_eval': 0,
        'estimation_method': 'structure_based'
    }
    
    total_flops = 0
    expert_flops = 0
    router_flops = 0
    
    # 각 컴포넌트별로 FLOPs 추정
    for name, module in neck.named_children():
        module_flops = 0
        
        # Linear layers
        if isinstance(module, nn.Linear):
            # FLOPs = 2 * input_dim * output_dim * batch * seq_len
            # (2는 곱셈과 덧셈)
            module_flops = 2 * module.in_features * module.out_features * seq_len
        
        # 서브모듈 재귀적으로 계산
        for sub_name, sub_module in module.named_modules():
            if isinstance(sub_module, nn.Linear):
                module_flops += 2 * sub_module.in_features * sub_module.out_features * seq_len
            elif isinstance(sub_module, nn.MultiheadAttention):
                # Attention FLOPs = 4 * seq_len * dim * dim + 2 * seq_len^2 * dim
                embed_dim = sub_module.embed_dim
                # Q, K, V projections
                module_flops += 3 * (2 * seq_len * embed_dim * embed_dim)
                # Attention scores: Q @ K^T
                module_flops += 2 * seq_len * seq_len * embed_dim
                # Attention output: Attn @ V
                module_flops += 2 * seq_len * seq_len * embed_dim
                # Output projection
                module_flops += 2 * seq_len * embed_dim * embed_dim
            elif isinstance(sub_module, nn.LayerNorm):
                # LayerNorm FLOPs = 2 * normalized_shape * seq_len
                if hasattr(sub_module, 'normalized_shape'):
                    norm_size = np.prod(sub_module.normalized_shape)
                    module_flops += 2 * norm_size * seq_len
            elif isinstance(sub_module, nn.Conv2d):
                # Conv2D FLOPs = 2 * kernel_h * kernel_w * in_ch * out_ch * out_h * out_w
                k_h, k_w = sub_module.kernel_size if isinstance(sub_module.kernel_size, tuple) else (sub_module.kernel_size, sub_module.kernel_size)
                module_flops += 2 * k_h * k_w * sub_module.in_channels * sub_module.out_channels * seq_len
        
        # Expert인지 확인 (ModuleList나 이름에 'expert'가 포함)
        if 'expert' in name.lower() and isinstance(module, nn.ModuleList):
            # Expert FLOPs는 별도로 저장 (활성화 비율 계산용)
            if len(module) > 0:
                single_expert_flops = sum(
                    2 * m.in_features * m.out_features * seq_len 
                    for m in module[0].modules() if isinstance(m, nn.Linear)
                )
                expert_flops = single_expert_flops * len(module)
                module_flops = expert_flops  # 일단 전체 expert flops 저장
        elif 'gate' in name.lower() or 'router' in name.lower():
            # Router FLOPs는 항상 사용됨
            router_flops = module_flops
        
        total_flops += module_flops
    
    # Position embedding 처리
    for param_name, param in neck.named_parameters(recurse=False):
        if 'pos_embed' in param_name:
            # Position embedding addition: seq_len * embed_dim
            total_flops += param.numel()
    
    # MoE의 경우 활성 expert 비율 고려
    if num_experts and train_top_k and eval_top_k and expert_flops > 0:
        single_expert_flops = expert_flops / num_experts
        
        # Router + 활성 experts
        train_active_flops = router_flops + single_expert_flops * train_top_k
        eval_active_flops = router_flops + single_expert_flops * eval_top_k
        
        # 나머지 컴포넌트 (pos_embed 등) 추가
        other_flops = total_flops - expert_flops - router_flops
        
        result['neck_train'] = int(train_active_flops + other_flops)
        result['neck_eval'] = int(eval_active_flops + other_flops)
        result['expert_flops'] = int(single_expert_flops)
        result['router_flops'] = int(router_flops)
    else:
        # MoE가 아닌 경우 모든 파라미터 사용
        result['neck_train'] = int(total_flops)
        result['neck_eval'] = int(total_flops)
    
    return result


def analyze_moe_flops(neck: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, int]:
    """MoE Neck의 Train/Eval 모드별 FLOPs를 추정합니다 (일반적인 방식)."""
    flops_info = {
        'neck_train': 0,
        'neck_eval': 0,
        'estimation_method': 'parameter_based'
    }
    
    # 입력 크기로부터 feature map 크기 추정
    C, H, W = input_shape
    seq_len = H * W
    
    # Neck의 top_k 정보 찾기
    train_top_k = None
    eval_top_k = None
    num_experts = None
    
    for name, module in neck.named_modules():
        if hasattr(module, 'top_k'):
            train_top_k = module.top_k
        if hasattr(module, 'eval_top_k'):
            eval_top_k = module.eval_top_k
        if hasattr(module, 'num_experts'):
            num_experts = module.num_experts
    
    # thop으로 시도
    try:
        from thop import profile
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        neck_copy = neck.to(device)
        neck_copy.eval()
        
        dummy_input = torch.randn(1, C, H, W).to(device)
        
        # Eval 모드 FLOPs
        with torch.no_grad():
            eval_flops, _ = profile(neck_copy, inputs=(dummy_input,), verbose=False)
        
        flops_info['neck_eval'] = int(eval_flops)
        
        # Train 모드 FLOPs (MoE가 있고 top_k가 다른 경우)
        if train_top_k and eval_top_k and train_top_k != eval_top_k:
            # top_k 비율로 추정
            ratio = train_top_k / eval_top_k if eval_top_k > 0 else 1
            flops_info['neck_train'] = int(eval_flops * ratio)
        else:
            flops_info['neck_train'] = flops_info['neck_eval']
        
        flops_info['estimation_method'] = 'thop'
        
    except Exception as e:
        # thop 실패 시 파라미터 기반 추정
        print(f"  💡 FLOPs 직접 계산 실패, 파라미터 기반 추정 사용: {e}")
        
        # 더 정확한 FLOPs 추정
        flops_info.update(estimate_neck_flops_from_structure(
            neck, seq_len, train_top_k, eval_top_k, num_experts
        ))
    
    # MoE 정보 추가
    if train_top_k is not None:
        flops_info['train_top_k'] = train_top_k
    if eval_top_k is not None:
        flops_info['eval_top_k'] = eval_top_k
    if num_experts is not None:
        flops_info['num_experts'] = num_experts
    
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
    
    # Neck 분석
    if hasattr(model, 'neck') and model.neck is not None:
        print(f"\n🔗 Neck: {type(model.neck).__name__}")
        print("-" * 40)
        try:
            actual_params = sum(p.numel() for p in model.neck.parameters())
            print(f"  🔢 파라미터:         {format_number(actual_params):>12}")
            
            # FLOPs 분석
            moe_flops = analyze_moe_flops(model.neck, backbone_output_shape)
            
            print(f"  📊 추정 방법:        {moe_flops.get('estimation_method', 'unknown')}")
            print(f"  ⚡ Train FLOPs:      {format_number(moe_flops['neck_train']):>12}")
            print(f"  ⚡ Eval FLOPs:       {format_number(moe_flops['neck_eval']):>12}")
            
            # 세부 FLOPs 정보 (structure_based 추정인 경우)
            if moe_flops.get('estimation_method') == 'structure_based':
                if 'router_flops' in moe_flops:
                    print(f"\n  📊 세부 FLOPs:")
                    print(f"     ├─ Router:          {format_number(moe_flops['router_flops']):>12}")
                    if 'expert_flops' in moe_flops:
                        print(f"     └─ Single Expert:   {format_number(moe_flops['expert_flops']):>12}")
            
            # MoE 정보가 있으면 출력
            if 'train_top_k' in moe_flops or 'eval_top_k' in moe_flops:
                print(f"\n  💡 MoE 설정:")
                if 'num_experts' in moe_flops:
                    print(f"     ├─ Experts: {moe_flops['num_experts']}")
                if 'train_top_k' in moe_flops:
                    print(f"     ├─ Train top-k: {moe_flops['train_top_k']}")
                if 'eval_top_k' in moe_flops:
                    print(f"     └─ Eval top-k: {moe_flops['eval_top_k']}")
            
            neck_train_flops = moe_flops['neck_train']
            neck_eval_flops = moe_flops['neck_eval']
            total_params += actual_params
            
        except Exception as e:
            print(f"  ❌ FLOPs 계산 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # Head 분석 (ETFHead는 파라미터가 없으므로 생략)
    # ETFHead는 고정된 ETF classifier로 학습 가능한 파라미터가 없음

    if neck_train_flops > 0 or neck_eval_flops > 0:
        print(f"\n🏆 Neck FLOPs 요약")
        print("=" * 60)
        
        print(f"  📊 Train FLOPs (Neck): {format_number(neck_train_flops):>12}")
        print(f"  📊 Eval FLOPs (Neck):  {format_number(neck_eval_flops):>12}")
        
        if neck_train_flops != neck_eval_flops and neck_eval_flops > 0:
            flops_reduction = neck_train_flops - neck_eval_flops
            reduction_ratio = (flops_reduction / neck_train_flops * 100) if neck_train_flops > 0 else 0
            print(f"  💡 Eval FLOPs 절감:   {format_number(flops_reduction):>12} ({reduction_ratio:.1f}% 감소)")
        
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


def get_module_tree(module: nn.Module, max_depth: int = 5, current_depth: int = 0) -> Dict[str, Any]:
    """모듈을 재귀적으로 분석하여 트리 구조로 반환합니다."""
    if current_depth >= max_depth:
        return None
    
    # 직속 파라미터 (서브모듈 제외)
    direct_params = sum(p.numel() for p in module.parameters(recurse=False))
    total_params = sum(p.numel() for p in module.parameters())
    
    result = {
        'total_params': total_params,
        'direct_params': direct_params,
        'type': type(module).__name__,
        'children': {}
    }
    
    # 직속 서브모듈 분석
    for name, child_module in module.named_children():
        child_params = sum(p.numel() for p in child_module.parameters())
        if child_params > 0:
            child_tree = get_module_tree(child_module, max_depth, current_depth + 1)
            if child_tree:
                result['children'][name] = child_tree
    
    return result


def print_module_tree(tree: Dict[str, Any], name: str = "root", indent: int = 0, is_last: bool = True, prefix: str = ""):
    """모듈 트리를 보기 좋게 출력합니다."""
    if tree is None:
        return
    
    # 트리 구조 문자
    if indent == 0:
        connector = ""
        next_prefix = ""
    else:
        connector = "└─ " if is_last else "├─ "
        next_prefix = prefix + ("   " if is_last else "│  ")
    
    # 현재 노드 출력
    total = tree['total_params']
    direct = tree['direct_params']
    type_name = tree['type']
    
    if direct > 0:
        print(f"{prefix}{connector}{name:30} {format_number(total):>12} (직접: {format_number(direct):>10}) [{type_name}]")
    else:
        print(f"{prefix}{connector}{name:30} {format_number(total):>12} [{type_name}]")
    
    # 자식 노드 재귀 출력
    children = list(tree['children'].items())
    for i, (child_name, child_tree) in enumerate(children):
        is_last_child = (i == len(children) - 1)
        print_module_tree(child_tree, child_name, indent + 1, is_last_child, next_prefix)


def analyze_moe_neck(neck: nn.Module) -> Dict[str, Any]:
    """MoE Neck의 세부 구조를 분석합니다 (재귀적으로 최소 단위까지)."""
    stats = {
        'total_params': sum(p.numel() for p in neck.parameters()),
        'tree': get_module_tree(neck, max_depth=5)
    }
    
    # MoE 관련 정보가 있으면 추가
    moe_info = {}
    for name, module in neck.named_modules():
        if hasattr(module, 'top_k'):
            moe_info['train_top_k'] = module.top_k
        if hasattr(module, 'eval_top_k'):
            moe_info['eval_top_k'] = module.eval_top_k
        if hasattr(module, 'num_experts'):
            moe_info['num_experts'] = module.num_experts
    
    if moe_info:
        stats['moe_info'] = moe_info
    
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
        total_str = f"{format_number(stats['total']):>12}"
        trainable_str = f"{format_number(stats['trainable']):>12}"
        frozen_str = f"{format_number(stats['frozen']):>12}"
        print(f"  {comp_name:12}: {total_str} (훈련: {trainable_str} / 고정: {frozen_str})")
    
    # MoE Neck 세부 분석
    if hasattr(model, 'neck') and model.neck is not None:
        moe_stats = analyze_moe_neck(model.neck)
        
        print(f"\n🔀 Neck 세부 분석 (재귀 트리 구조):")
        print("-" * 80)
        print(f"  총 Neck 파라미터: {format_number(moe_stats['total_params']):>12}\n")
        
        # 트리 구조 출력
        if 'tree' in moe_stats and moe_stats['tree']:
            print(f"  📦 모듈 계층 구조:")
            print()
            # Neck 자체 출력
            tree = moe_stats['tree']
            print(f"  Neck                           {format_number(tree['total_params']):>12} [{tree['type']}]")
            # 자식들 출력
            children = list(tree['children'].items())
            for i, (child_name, child_tree) in enumerate(children):
                is_last = (i == len(children) - 1)
                print_module_tree(child_tree, child_name, indent=1, is_last=is_last, prefix="  ")
        
        # MoE 정보 출력 (있는 경우)
        if 'moe_info' in moe_stats:
            info = moe_stats['moe_info']
            print(f"\n  ⚡ MoE 설정:")
            if 'num_experts' in info:
                print(f"     ├─ Expert 개수:     {info['num_experts']}")
            if 'train_top_k' in info:
                print(f"     ├─ Train Top-K:     {info['train_top_k']}")
            if 'eval_top_k' in info:
                print(f"     └─ Eval Top-K:      {info['eval_top_k']}")
            
            # 활성화 비율 계산 (가능한 경우)
            if all(k in info for k in ['num_experts', 'train_top_k', 'eval_top_k']):
                train_ratio = info['train_top_k'] / info['num_experts'] * 100
                eval_ratio = info['eval_top_k'] / info['num_experts'] * 100
                print(f"\n  💡 Expert 활성화 비율:")
                print(f"     ├─ Train: {train_ratio:.1f}% ({info['train_top_k']}/{info['num_experts']})")
                print(f"     └─ Eval:  {eval_ratio:.1f}% ({info['eval_top_k']}/{info['num_experts']})")
    
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
