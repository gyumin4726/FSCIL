_base_ = [
    '../_base_/models/vmamba_etf.py', '../_base_/datasets/cifar_fscil.py',
    '../_base_/schedules/cifar_200e.py', '../_base_/default_runtime.py'
]

# CIFAR requires different inc settings
inc_start = 60
inc_end = 100
inc_step = 5

model = dict(backbone=dict(type='VMambaBackbone',
                           model_name='vmamba_base_s2l15',  # 모델 변경
                           pretrained_path='./vssm_base_0229_ckpt_epoch_237.pth',
                           out_indices=(0, 1, 2, 3),  # Extract features from all 4 stages
                           frozen_stages=0,  # Freeze patch embedding and first stage
                           channel_first=True),
             neck=dict(type='MambaNeck',
                       version='ss2d',
                       in_channels=1024,  # VMamba base stage4 output channels
                       out_channels=1024,
                       feat_size=1,  # 실제 출력 크기에 맞게 수정 (5×5)
                       num_layers=3,
                       use_residual_proj=True,
                       # Enhanced skip connection settings (MASC-M) for VMamba features
                       use_multi_scale_skip=True,
                       multi_scale_channels=[128, 256, 512]),
             head=dict(type='ETFHead',
                       in_channels=1024,
                       num_classes=100,
                       eval_classes=60,
                       with_len=False,
                       cal_acc=True,
                       loss=dict(type='CombinedLoss', dr_weight=0.0, ce_weight=1.0)),
             mixup=0,
             mixup_prob=0)
