_base_ = [
    '../_base_/models/vmamba_etf.py', '../_base_/datasets/cub_fscil.py',
    '../_base_/schedules/cub_20e.py', '../_base_/default_runtime.py'
]

inc_start = 100
inc_end = 200
inc_step = 10

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
                      eval_top_k=2,
                      in_channels=1024, 
                      out_channels=1024,
                      feat_size=9,
                      use_aux_loss=False,
                      aux_loss_weight=0.01),
             head=dict(type='ETFHead',
                       in_channels=1024,
                       num_classes=200,
                       eval_classes=100,
                       with_len=False,
                       cal_acc=True,
                       loss=dict(type='CombinedLoss', dr_weight=1.0, ce_weight=0.0)),
             mixup=0,
             mixup_prob=0)

optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'neck.pos_embed': dict(lr_mult=1.0),
            'neck.moe.gate': dict(lr_mult=1.5),     
            'neck.moe.experts': dict(lr_mult=5.0),
        }
    ))

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealingCooldown',
    min_lr=None,
    min_lr_ratio=0.1,
    cool_down_ratio=0.1,
    cool_down_time=10,
    by_epoch=False,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=20)

find_unused_parameters = True