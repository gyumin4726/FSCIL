_base_ = [
    '../_base_/models/vmamba_etf.py', '../_base_/datasets/cifar_fscil.py',
    '../_base_/schedules/cifar_200e.py', '../_base_/default_runtime.py'
]

# CIFAR requires different inc settings
inc_start = 60
inc_end = 100
inc_step = 5

# model settings
model = dict(backbone=dict(type='VMambaBackbone',
                           model_name='vmamba_base_s2l15',  # 모델 변경
                           pretrained_path='./vssm_base_0229_ckpt_epoch_237.pth',
                           out_indices=(0, 1, 2, 3),  # Multi-scale features from all stages
                           frozen_stages=0,  # Freeze patch embedding and first stage
                           channel_first=True),
             neck=dict(type='MambaNeck',
                       version='ss2d',
                       in_channels=1024,  # VMamba base stage4 channels
                       out_channels=1024,
                       feat_size=1,  # 실제 출력 크기에 맞게 수정 (5×5)
                       num_layers=3,
                       use_residual_proj=True,
                       use_new_branch=True,
                       detach_residual=False,
                       num_layers_new=3,
                       loss_weight_supp=100,
                       loss_weight_supp_novel=10,
                       loss_weight_sep=0.001,
                       loss_weight_sep_new=0.001,
                       param_avg_dim='0-1-3',
                       # Enhanced skip connection settings (MASC-M)
                       use_multi_scale_skip=False,
                       multi_scale_channels=[128, 256, 512]),
             head=dict(type='ETFHead',
                       in_channels=1024,
                       num_classes=100,
                       eval_classes=60,
                       loss=dict(type='CombinedLoss', dr_weight=0.0, ce_weight=1.0),
                       with_len=False),
             mixup=0.5,
             mixup_prob=0.75)

copy_list = (1, 1, 1, 1, 1, 1, 1, 1, None, None)
step_list = (200, 200, 200, 200, 200, 200, 200, 200, None, None)

finetune_lr = 0.25

# optimizer
optimizer = dict(type='SGD',
                 lr=finetune_lr,
                 momentum=0.9,
                 weight_decay=0.0005,
                 paramwise_cfg=dict(
                     custom_keys={
                         'neck.mlp_proj.': dict(lr_mult=0.2),
                         'neck.block.': dict(lr_mult=0.2),
                         'neck.residual_proj': dict(lr_mult=0.2),
                         'neck.pos_embed': dict(lr_mult=0.2),
                         'neck.pos_embed_new': dict(lr_mult=1),
                         # Enhanced skip connection components
                         'neck.multi_scale_adapters': dict(lr_mult=0.5),
                         'neck.skip_attention': dict(lr_mult=1.0),
                         'neck.skip_proj': dict(lr_mult=1.0),
                     }))

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

find_unused_parameters=True