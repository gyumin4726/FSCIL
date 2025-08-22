_base_ = [
    '../_base_/models/vmamba_etf.py', '../_base_/datasets/cub_fscil.py',
    '../_base_/schedules/cub_80e.py', '../_base_/default_runtime.py'
]

# CUB requires different inc settings
inc_start = 100
inc_end = 200
inc_step = 10

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
                       feat_size=7,  # 224 / (4*8) = 7 (patch_size=4, 4 downsample stages with 2x each)
                       num_layers=3,
                       use_residual_proj=True,
                       # Enhanced skip connection settings (MASC-M) for VMamba features
                       use_multi_scale_skip=True,
                       multi_scale_channels=[128, 256, 512]),
             head=dict(type='ETFHead',
                       in_channels=1024,
                       num_classes=200,
                       eval_classes=100,
                       with_len=False,
                       cal_acc=True,
                       loss=dict(type='CombinedLoss', dr_weight=0.0, ce_weight=1.0)),
             mixup=0,
             mixup_prob=0)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.2,
    momentum=0.9,
    weight_decay=0.0005,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealingCooldown',
    min_lr=None,
    min_lr_ratio=0.1,
    cool_down_ratio=0.1,
    cool_down_time=10,
    by_epoch=False,
    # warmup
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=20) 