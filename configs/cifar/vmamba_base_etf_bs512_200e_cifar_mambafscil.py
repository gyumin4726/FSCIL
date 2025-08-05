_base_ = [
    '../_base_/models/vmamba_etf.py', '../_base_/datasets/cifar_fscil.py',
    '../_base_/schedules/cifar_200e.py', '../_base_/default_runtime.py'
]

# CIFAR requires different inc settings
inc_start = 60
inc_end = 100
inc_step = 5

model = dict(backbone=dict(_delete_=True,
                           type='VMambaBackbone',
                           model_name='vmamba_base_s2l15',  # 모델 변경
                           pretrained_path='./vssm_base_0229_ckpt_epoch_237.pth',
                           out_indices=(0, 1, 2, 3),  # Extract features from all 4 stages
                           frozen_stages=0,  # Freeze patch embedding and first stage
                           channel_first=True),
             neck=dict(type='MambaNeck',
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
                       num_classes=100,
                       eval_classes=60,
                       with_len=False,
                       cal_acc=True,
                       loss=dict(type='CombinedLoss', dr_weight=0.0, ce_weight=1.0)),
             mixup=0,
             mixup_prob=0)

img_size = 32
_img_resize_size = 36
img_norm_cfg = dict(mean=[129.304, 124.070, 112.434],
                    std=[68.170, 65.392, 70.418],
                    to_rgb=False)
meta_keys = ('filename', 'ori_filename', 'ori_shape', 'img_shape', 'flip',
             'flip_direction', 'img_norm_cfg', 'cls_id', 'img_id')

train_pipeline = [
    dict(type='RandomResizedCrop',
         size=img_size,
         scale=(0.6, 1.),
         interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

test_pipeline = [
    dict(type='Resize', size=(_img_resize_size, -1), interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

data = dict(samples_per_gpu=64,
            workers_per_gpu=8,
            train_dataloader=dict(persistent_workers=True, ),
            val_dataloader=dict(persistent_workers=True, ),
            test_dataloader=dict(persistent_workers=True, ),
            train=dict(type='RepeatDataset',
                       times=1,
                       dataset=dict(
                           type='CIFAR100FSCILDataset',
                           data_prefix='./data/cifar',
                           pipeline=train_pipeline,
                           num_cls=60,
                           subset='train',
                       )),
            val=dict(
                type='CIFAR100FSCILDataset',
                data_prefix='./data/cifar',
                pipeline=test_pipeline,
                num_cls=60,
                subset='test',
            ),
            test=dict(
                type='CIFAR100FSCILDataset',
                data_prefix='./data/cifar',
                pipeline=test_pipeline,
                num_cls=100,
                subset='test',
            ))
