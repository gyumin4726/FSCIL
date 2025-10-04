_base_ = [
    '../_base_/models/vmamba_etf.py',
    '../_base_/datasets/mini_imagenet_fscil.py',
    '../_base_/schedules/mini_imagenet_500e.py', '../_base_/default_runtime.py'
]

model = dict(backbone=dict(type='VMambaBackbone',
                           model_name='vmamba_base_s2l15',
                           pretrained_path='./vssm_base_0229_ckpt_epoch_237.pth',
                           #model_name='vmamba_tiny_s1l8', 
                           #pretrained_path='./vssm1_tiny_0230s_ckpt_epoch_264.pth',
                           #model_name='vmamba_small_s2l15', 
                           #pretrained_path='./vssm_small_0229_ckpt_epoch_222.pth',                           
                           out_indices=(0, 1, 2, 3),
                           frozen_stages=0,
                           channel_first=True),
             neck=dict(type='MoEFSCILNeck',
                       num_experts=4,
                       top_k=4,
                       in_channels=1024,
                       out_channels=1024,
                       feat_size=3,
                       use_aux_loss=False,
                       aux_loss_weight=0.01),
             head=dict(type='ETFHead',
                       in_channels=1024,
                       loss=dict(type='CombinedLoss', dr_weight=1.0, ce_weight=0.0),
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
                         # 기본 컴포넌트들
                         'neck.pos_embed': dict(lr_mult=10.0),
                         
                         # MoE 컴포넌트들
                         'neck.moe.gate': dict(lr_mult=10.0),     
                         'neck.moe.experts': dict(lr_mult=10.0),  
                     }))

find_unused_parameters = True