_base_ = [
    '../_base_/models/vmamba_etf.py', '../_base_/datasets/cifar_fscil.py',
    '../_base_/schedules/cifar_200e.py', '../_base_/default_runtime.py'
]

inc_start = 60
inc_end = 100
inc_step = 5

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
                       top_k=2,
                       in_channels=1024,
                       out_channels=1024,
                       feat_size=1,
                       use_multi_scale_skip=True,
                       multi_scale_channels=[128, 256, 512],
                       d_state=16,
                       dt_rank=64,
                       ssm_expand_ratio=1.0,
                       num_heads=8,
                       use_aux_loss=False,
                       aux_loss_weight=0.01),
             head=dict(type='ETFHead',
                       in_channels=1024,
                       num_classes=100,
                       eval_classes=60,
                       with_len=False,
                       cal_acc=True,
                       loss=dict(type='CombinedLoss', dr_weight=1.0, ce_weight=0.0)),
             mixup=0,
             mixup_prob=0)

optimizer = dict(
    type='SGD',
    lr=0.25,
    momentum=0.9,
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys={
            # 기본 컴포넌트들
            'backbone': dict(lr_mult=0.1),
            'neck.pos_embed': dict(lr_mult=1.0),
            
            # MoE 컴포넌트들
            'neck.moe.gate': dict(lr_mult=1.5),     
            'neck.moe.experts': dict(lr_mult=5.0),
            
            # Multi-Scale 관련 (use_multi_scale_skip=True일 때만 사용됨)
            'neck.multi_scale_adapters': dict(lr_mult=1.0),
            'neck.multi_scale_router.spatial_self_attention': dict(lr_mult=1.0),
            'neck.multi_scale_router.aux_layer_cross_attention': dict(lr_mult=1.0),
        }
    ))

find_unused_parameters = True