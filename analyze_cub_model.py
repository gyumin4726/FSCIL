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


def analyze_components_flops(model, input_shape: Tuple[int, ...] = (3, 224, 224)):
    """모델의 각 컴포넌트별로 FLOPs를 분석합니다."""
    print(f"\n⚡ 컴포넌트별 FLOPs 분석 (입력 크기: {input_shape})")
    print("=" * 60)
    
    total_flops = 0
    total_params = 0
    
    # Backbone 분석
    if hasattr(model, 'backbone') and model.backbone is not None:
        print(f"\n🔧 Backbone: {type(model.backbone).__name__}")
        print("-" * 40)
        backbone_flops, thop_params = try_thop_flops_component(model.backbone, input_shape, "Backbone")
        # 실제 파라미터 수는 직접 계산 (thop이 놓칠 수 있음)
        actual_params = sum(p.numel() for p in model.backbone.parameters())
        
        if backbone_flops is not None:
            print(f"  📊 FLOPs:      {format_number(backbone_flops):>12}")
            print(f"  🔢 파라미터:    {format_number(actual_params):>12}")
            total_flops += backbone_flops
            total_params += actual_params
        else:
            print("  ❌ FLOPs 계산 실패")
            print(f"  🔢 파라미터:    {format_number(actual_params):>12}")
            total_params += actual_params
        
        # Backbone 출력 크기 자동 감지
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
                print(f"  ✅ Backbone 출력 크기: {backbone_output_shape}")
        except Exception as e:
            # 실패하면 neck의 in_channels 사용
            if hasattr(model.neck, 'in_channels'):
                backbone_output_shape = (model.neck.in_channels, 7, 7)
                print(f"  💡 Neck의 in_channels로부터 추정: {backbone_output_shape}")
            else:
                backbone_output_shape = (1024, 7, 7)  # 기본값
                print(f"  ⚠️ 기본값 사용: {backbone_output_shape}")
    
    # Neck 분석
    if hasattr(model, 'neck') and model.neck is not None:
        print(f"\n🔗 Neck: {type(model.neck).__name__}")
        print("-" * 40)
        try:
            # Neck은 backbone의 출력을 입력으로 받음
            neck_input_shape = backbone_output_shape
            neck_flops, thop_params = try_thop_flops_component(model.neck, neck_input_shape, "Neck")
            # 실제 파라미터 수는 직접 계산 (thop이 놓칠 수 있음)
            actual_params = sum(p.numel() for p in model.neck.parameters())
            
            if neck_flops is not None:
                print(f"  📊 FLOPs:      {format_number(neck_flops):>12}")
                print(f"  🔢 파라미터:    {format_number(actual_params):>12}")
                total_flops += neck_flops
                total_params += actual_params
            else:
                print("  ❌ FLOPs 계산 실패")
                print(f"  🔢 파라미터:    {format_number(actual_params):>12}")
                total_params += actual_params
        except Exception as e:
            print(f"  ❌ FLOPs 계산 실패: {e}")
    
    # Head 분석
    if hasattr(model, 'head') and model.head is not None:
        print(f"\n🎯 Head: {type(model.head).__name__}")
        print("-" * 40)
        try:
            # Head는 neck의 출력을 입력으로 받음 (일반적으로 1D feature)
            # neck의 out_channels 또는 head의 in_channels 사용
            if hasattr(model.neck, 'out_channels'):
                head_input_dim = model.neck.out_channels
            elif hasattr(model.head, 'in_channels'):
                head_input_dim = model.head.in_channels
            else:
                head_input_dim = 1024  # 기본값
            head_input_shape = (head_input_dim,)  # Neck 출력 차원
            head_flops, thop_params = try_thop_flops_component(model.head, head_input_shape, "Head")
            # 실제 파라미터 수는 직접 계산 (thop이 놓칠 수 있음)
            actual_params = sum(p.numel() for p in model.head.parameters())
            
            if head_flops is not None:
                print(f"  📊 FLOPs:      {format_number(head_flops):>12}")
                print(f"  🔢 파라미터:    {format_number(actual_params):>12}")
                total_flops += head_flops
                total_params += actual_params
            else:
                print("  ❌ FLOPs 계산 실패")
                print(f"  🔢 파라미터:    {format_number(actual_params):>12}")
                total_params += actual_params
        except Exception as e:
            print(f"  ❌ FLOPs 계산 실패: {e}")
    
    if total_flops > 0:
        print(f"\n🏆 전체 추정값")
        print("=" * 40)
        print(f"  📊 총 FLOPs:      {format_number(total_flops):>12}")
        print(f"  🔢 총 파라미터:    {format_number(total_params):>12}")
    
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
