import os
import warnings
import torch
import torch.nn as nn
warnings.filterwarnings(
    "ignore",
    message=r".*MMCV will release v2\.0\.0.*",
    category=UserWarning
)
from mmcv import Config
from mmcls.models import build_classifier
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

def analyze_moe_flops(neck: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, int]:
    """MoE Neck FLOPs를 thop으로만 계산합니다."""
    C, H, W = input_shape

    result: Dict[str, int] = {}
    try:
        from thop import profile
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        neck_copy = neck.to(device)
        neck_copy.eval()
        dummy_input = torch.randn(1, C, H, W).to(device)
        with torch.no_grad():
            eval_flops, _ = profile(neck_copy, inputs=(dummy_input,), verbose=False)
        thop_eval = int(eval_flops)
        result['neck_train'] = thop_eval
        result['neck_eval'] = thop_eval
        result['estimation_method'] = 'thop'
    except Exception as e:
        print(f"  ❌ thop FLOPs 계산 실패: {e}")
        result['estimation_method'] = 'error'

    return result


def analyze_components_flops(model, input_shape: Tuple[int, ...] = (3, 224, 224)):
    """모델의 각 컴포넌트별로 FLOPs를 분석합니다."""
    print(f"\n⚡ 컴포넌트별 FLOPs 분석 (입력 크기: {input_shape})")
    print("=" * 60)
    
    if hasattr(model, 'neck') and model.neck is not None:
        c = getattr(model.neck, 'in_channels', 1024)
        f = getattr(model.neck, 'feat_size', 7)
        backbone_output_shape = (c, f, f)
    else:
        backbone_output_shape = (1024, 7, 7)
    
    # Neck 분석
    if hasattr(model, 'neck') and model.neck is not None:
        print(f"\n🔗 Neck: {type(model.neck).__name__}")
        print("-" * 40)
        try:
            actual_params = sum(p.numel() for p in model.neck.parameters())
            print(f"  🔢 파라미터:         {format_number(actual_params):>12}")
            
            # FLOPs 분석 (thop)
            moe_flops = analyze_moe_flops(model.neck, backbone_output_shape)
            
            method = moe_flops.get('estimation_method', 'unknown')
            print(f"  📊 추정 방법:        {method}")
            if 'neck_train' in moe_flops and 'neck_eval' in moe_flops:
                print(f"  ⚡ Train FLOPs:      {format_number(moe_flops['neck_train']):>12}")
                print(f"  ⚡ Eval  FLOPs:      {format_number(moe_flops['neck_eval']):>12}")
            
        except Exception as e:
            print(f"  ❌ FLOPs 계산 실패: {e}")
            import traceback
            traceback.print_exc()
    
    return None


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

def get_module_tree(module: nn.Module, max_depth: int = 12, current_depth: int = 0) -> Dict[str, Any]:
    """모듈을 재귀적으로 분석하여 트리 구조로 반환합니다."""
    if current_depth >= max_depth:
        return None
    
    # 직속 파라미터/버퍼 수집 (서브모듈 제외)
    direct_params = sum(p.numel() for p in module.parameters(recurse=False))
    direct_param_items = {name: p.numel() for name, p in module.named_parameters(recurse=False)}
    direct_buffer_items = {name: b.numel() for name, b in module.named_buffers(recurse=False)}
    
    total_params = sum(p.numel() for p in module.parameters())
    
    result = {
        'total_params': total_params,
        'direct_params': direct_params,
        'direct_param_items': direct_param_items,
        'direct_buffer_items': direct_buffer_items,
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
    
    # 이 노드의 직속 파라미터/버퍼 이름별 표기
    param_items = tree.get('direct_param_items', {})
    buffer_items = tree.get('direct_buffer_items', {})
    if param_items:
        for i, (pname, psize) in enumerate(param_items.items()):
            line_connector = "└─ " if (not buffer_items and i == len(param_items) - 1 and not tree['children']) else "├─ "
            print(f"{next_prefix}{line_connector}param {pname:22} {format_number(psize):>12}")
    if buffer_items:
        last_idx = len(buffer_items) - 1
        for i, (bname, bsize) in enumerate(buffer_items.items()):
            line_connector = "└─ " if (i == last_idx and not tree['children']) else "├─ "
            print(f"{next_prefix}{line_connector}buffer {bname:20} {format_number(bsize):>12}")
    
    # 자식 노드 재귀 출력
    children = list(tree['children'].items())
    # 전문가 목록(ModuleList) 축약 출력: 동일 구조라면 첫 번째만 상세, 나머지는 생략 안내
    if (name.lower().endswith('experts') or tree.get('type') == 'ModuleList') and len(children) > 1:
        # 동일성 판단: type/total_params가 모두 동일한지 확인
        first_child = children[0][1]
        homogeneous = all(
            (ct['type'] == first_child['type'] and ct['total_params'] == first_child['total_params'])
            for _, ct in children
        )
        if homogeneous:
            # 첫 번째 expert만 출력
            print_module_tree(first_child, children[0][0], indent + 1, not children[1:], next_prefix)
            # 생략 안내 라인
            omitted = len(children) - 1
            connector2 = "└─ "
            print(f"{next_prefix}{connector2}(동일 expert {omitted}개 생략)")
            return
    
    for i, (child_name, child_tree) in enumerate(children):
        is_last_child = (i == len(children) - 1)
        print_module_tree(child_tree, child_name, indent + 1, is_last_child, next_prefix)


def analyze_moe_neck(neck: nn.Module) -> Dict[str, Any]:
    """MoE Neck의 세부 구조를 분석합니다 (재귀적으로 최소 단위까지)."""
    stats = {
        'total_params': sum(p.numel() for p in neck.parameters()),
        'tree': get_module_tree(neck, max_depth=12)
    }
    
    return stats


def print_model_analysis(model: nn.Module, config_name: str, input_shape: Tuple[int, ...] = (3, 224, 224)):
    """모델 분석 결과를 출력합니다."""
    print("=" * 80)
    print(f"🔍 CUB 모델 분석 결과 - {config_name}")
    print("=" * 80)
    
    # 컴포넌트별 분석 (Neck만 표시)
    print(f"\n🧩 컴포넌트별 파라미터 (Neck만 표시):")
    print("-" * 50)
    if hasattr(model, 'neck') and model.neck is not None:
        neck_stats = count_parameters(model.neck)
        total_str = f"{format_number(neck_stats['total']):>12}"
        trainable_str = f"{format_number(neck_stats['trainable']):>12}"
        frozen_str = f"{format_number(neck_stats['frozen']):>12}"
        print(f"  neck        : {total_str} (훈련: {trainable_str} / 고정: {frozen_str})")
    
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
            # 루트(Neck)부터 전체 트리를 출력 (직속 파라미터/버퍼 포함)
            tree = moe_stats['tree']
            print_module_tree(tree, name="Neck", indent=0, is_last=True, prefix="  ")
    
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
