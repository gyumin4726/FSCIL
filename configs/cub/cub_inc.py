_base_ = [
    '../_base_/models/vmamba_etf.py', '../_base_/datasets/cub_fscil.py',
    '../_base_/schedules/cub_80e.py', '../_base_/default_runtime.py'
]

inc_start = 100
inc_end = 200
inc_step = 10

model = dict(backbone=dict(type='VMambaBackbone',
                           model_name='vmamba_base_s2l15',
                           pretrained_path='./vssm_base_0229_ckpt_epoch_237.pth',
                           out_indices=(0, 1, 2, 3),
                           frozen_stages=0,
                           channel_first=True),
             neck=dict(type='MoEFSCILNeck',
                       num_experts=8,
                       top_k=4,
                       in_channels=1024,
                       out_channels=1024,
                       feat_size=7,
                       d_state=256,
                       dt_rank=256,
                       ssm_expand_ratio=1.0,
                       num_heads=8,
                       use_aux_loss=False,
                       aux_loss_weight=0.01),
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
                         # 기본 MoE 컴포넌트들
                         'neck.mlp_proj': dict(lr_mult=0.8),
                         'neck.pos_embed': dict(lr_mult=0.8),
                         'neck.moe.gate': dict(lr_mult=0.8),     
                         'neck.moe.experts': dict(lr_mult=0.8),
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