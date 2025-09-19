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
             neck=dict(type='MoEFSCILNeck',
                       num_experts=4, 
                       top_k=1,  
                       in_channels=1024,
                       out_channels=1024,
                       feat_size=1,
                       num_layers=3,
                       d_state=256,
                       dt_rank=256,
                       ssm_expand_ratio=1.0,
                       use_aux_loss=True,
                       aux_loss_weight=0.01),
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
                         'neck.mlp_proj': dict(lr_mult=10.0),
                         'neck.pos_embed': dict(lr_mult=10.0),
                         'neck.moe.gate': dict(lr_mult=10.0),     
                         'neck.moe.experts': dict(lr_mult=10.0),  
                     }))

find_unused_parameters=True