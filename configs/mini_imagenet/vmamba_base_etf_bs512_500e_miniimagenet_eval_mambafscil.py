_base_ = [
    '../_base_/models/vmamba_etf.py',
    '../_base_/datasets/mini_imagenet_fscil.py',
    '../_base_/schedules/mini_imagenet_500e.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(type='VMambaBackbone',
                           model_name='vmamba_base_s2l15',  # 모델 변경
                           pretrained_path='./vssm_base_0229_ckpt_epoch_237.pth',
                           out_indices=(0, 1, 2, 3),  # Multi-scale features from all stages
                           frozen_stages=0,  # Freeze patch embedding and first stage
                           channel_first=True),
             neck=dict(type='MambaNeck',
                       version='ssm',
                       in_channels=1024,  # VMamba base stage4 channels
                       out_channels=1024,
                       feat_size=3,
                       num_layers=2,
                       use_residual_proj=True,
                       use_new_branch=True,
                       detach_residual=True,
                       num_layers_new=2,
                       loss_weight_supp=100,
                       loss_weight_sep_new=0.5,
                       param_avg_dim='0-1-3',
                       # Enhanced skip connection settings (MASC-M)
                       use_multi_scale_skip=False,
                       multi_scale_channels=[128, 256, 512]),
             head=dict(type='ETFHead',
                       in_channels=1024,
                       loss=dict(type='CombinedLoss', dr_weight=0.0, ce_weight=1.0),
                       with_len=True),
             mixup=0.5,
             mixup_prob=0.3)

base_copy_list = (1, 2, 3, 4, 5, 6, 7, 8, None, None)
step_list = (100, 110, 120, 130, 140, 150, 160, 170, None, None)
copy_list = (10, 10, 10, 10, 10, 10, 10, 10, None, None)

finetune_lr = 0.01

# optimizer
optimizer = dict(type='SGD',
                 lr=finetune_lr,
                 momentum=0.9,
                 weight_decay=0.0005,
                 paramwise_cfg=dict(
                     custom_keys={
                         'neck.block.': dict(lr_mult=10.0),
                         'neck.residual_proj': dict(lr_mult=10.0),
                         'neck.pos_embed': dict(lr_mult=10.0),
                         'neck.mlp_proj.': dict(lr_mult=10.0),
                         'neck.pos_embed_new': dict(lr_mult=10),
                         # Enhanced skip connection components
                         'neck.multi_scale_adapters': dict(lr_mult=10.0),
                         'neck.skip_attention': dict(lr_mult=10.0),
                         'neck.skip_proj': dict(lr_mult=10.0),
                     }))