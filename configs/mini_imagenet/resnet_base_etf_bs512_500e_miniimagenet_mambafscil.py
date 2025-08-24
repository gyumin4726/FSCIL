_base_ = [
    '../_base_/models/resnet_etf.py',
    '../_base_/datasets/mini_imagenet_fscil.py',
    '../_base_/schedules/mini_imagenet_500e.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(neck=dict(type='MambaNeck',
                       version='ssm',
                       in_channels=640,  # VMamba base stage4 output channels
                       out_channels=512,
                       feat_size=5,  # 224 / (4*8) = 7 (patch_size=4, 4 downsample stages with 2x each)
                       num_layers=2,
                       use_residual_proj=True,
                       # Enhanced skip connection settings (MASC-M) for VMamba features
                       use_multi_scale_skip=True,
                       multi_scale_channels=[64, 160, 320]),
             head=dict(type='ETFHead',
                       in_channels=512,
                       with_len=True,
                       loss=dict(type='CombinedLoss', dr_weight=0.0, ce_weight=1.0)),
             mixup=0,
             mixup_prob=0)