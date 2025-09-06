_base_ = [
    '../_base_/models/vmamba_etf.py', '../_base_/datasets/cub_fscil.py',
    '../_base_/schedules/cub_80e.py', '../_base_/default_runtime.py'
]

# CUB requires different inc settings
inc_start = 100
inc_end = 200
inc_step = 10

# model settings
model = dict(backbone=dict(type='VMambaBackbone',
                           model_name='vmamba_base_s2l15',
                           pretrained_path='./vssm_base_0229_ckpt_epoch_237.pth',
                           out_indices=(0, 1, 2, 3),
                           frozen_stages=0,
                           channel_first=True),
             neck=dict(type='MoEFSCILNeck',
                       num_experts=4,
                       top_k=2,
                       version='ss2d',
                       in_channels=1024,
                       out_channels=1024,
                       feat_size=7,
                       num_layers=3,
                       use_residual_proj=True,
                       d_state=256,
                       dt_rank=256,
                       ssm_expand_ratio=1.0,
                       loss_weight_supp=100,
                       loss_weight_supp_novel=10,
                       loss_weight_sep=0.001,
                       loss_weight_sep_new=0.001,
                       param_avg_dim='0-1-3'),
             head=dict(type='ETFHead',
                       in_channels=1024,
                       num_classes=200,
                       eval_classes=100,
                       loss=dict(type='CombinedLoss', dr_weight=1.0, ce_weight=0.0),
                       with_len=False),
             mixup=0.5,
             mixup_prob=0.5)

base_copy_list = (1, 1, 2, 2, 3, 3, 1, 1, 1, 1)
copy_list = (10, 10, 10, 10, 10, 10, 10, 10, 10, 10)
step_list = (200, 210, 220, 230, 240, 250, 260, 270, 280, 290)
finetune_lr = 0.05

# optimizer
optimizer = dict(type='SGD',
                 lr=finetune_lr,
                 momentum=0.9,
                 weight_decay=0.0005,
                 paramwise_cfg=dict(
                     custom_keys={
                         'neck.mlp_proj.': dict(lr_mult=1.2),
                         'neck.ssm_block.': dict(lr_mult=1.2),
                         'neck.residual_proj': dict(lr_mult=1.2),
                         'neck.pos_embed': dict(lr_mult=1.2),
                         'neck.moe.gate': dict(lr_mult=1.2),     
                         'neck.moe.experts': dict(lr_mult=1.2),  
                     }))

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

find_unused_parameters=True