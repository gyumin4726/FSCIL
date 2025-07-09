_base_ = [
    '../_base_/models/resnet_etf.py', '../_base_/datasets/cub_fscil.py',
    '../_base_/schedules/cub_80e.py', '../_base_/default_runtime.py'
]

# CUB requires different inc settings
inc_start = 100
inc_end = 200
inc_step = 10

model = dict(backbone=dict(_delete_=True,
                           type='ResNet',
                           depth=18,
                           frozen_stages=1,
                           out_indices=(0, 1, 2, 3),  # Layer1-4 모든 출력
                           init_cfg=dict(type='Pretrained',
                                         checkpoint='torchvision://resnet18'),
                           norm_cfg=dict(type='SyncBN', requires_grad=True)),
             neck=dict(type='MambaNeck',
                       version='ss2d',
                       in_channels=512,  # Layer4 채널
                       out_channels=1024,
                       feat_size=7,
                       num_layers=2,
                       use_residual_proj=True,
                       # Enhanced skip connection settings (MASC-M)
                       use_multi_scale_skip=True,
                       multi_scale_channels=[64, 128, 256]),  # Layer1-3 채널
             head=dict(type='ETFHead',
                       in_channels=1024,
                       num_classes=200,
                       eval_classes=100,
                       with_len=False,
                       cal_acc=True),
             mixup=0,
             mixup_prob=0)

# dataset settings
img_size = 224
_img_resize_size = 256
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
meta_keys = ('filename', 'ori_filename', 'ori_shape', 'img_shape', 'flip',
             'flip_direction', 'img_norm_cfg', 'cls_id', 'img_id')

# 원본 이미지용 파이프라인
original_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(_img_resize_size, _img_resize_size)),
    dict(type='CenterCrop', crop_size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

# 증강 이미지용 파이프라인
augmented_pipeline = [
    dict(type='LoadAugmentedImage',
         aug_dir='data/CUB_200_2011/augmented_images'),
    dict(type='Resize', size=(_img_resize_size, _img_resize_size)),
    dict(type='CenterCrop', crop_size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

# 회전된 이미지용 파이프라인 (별도 폴더)
rotated_pipeline = [
    dict(type='LoadRotatedImage', rot_dir='data/CUB_200_2011/rotated_images'),
    dict(type='Resize', size=(_img_resize_size, _img_resize_size)),
    dict(type='CenterCrop', crop_size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(_img_resize_size, -1)),
    dict(type='CenterCrop', crop_size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train_dataloader=dict(persistent_workers=True),
    val_dataloader=dict(persistent_workers=True),
    test_dataloader=dict(persistent_workers=True),
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(  # 원본 이미지
                type='CUBFSCILDataset',
                data_prefix='./data/CUB_200_2011',
                pipeline=original_pipeline,
                num_cls=100,
                subset='train',
            ),
            dict(  # 기존 증강 이미지
                type='CUBFSCILDataset',
                data_prefix='./data/CUB_200_2011',
                pipeline=augmented_pipeline,
                num_cls=100,
                subset='train',
            ),
            dict(  # 회전된 이미지 (모든 각도)
                type='CUBFSCILDataset',
                data_prefix='./data/CUB_200_2011',
                ann_file='./data/CUB_200_2011/rotated_train.txt',
                pipeline=rotated_pipeline,
                num_cls=100,
                subset='train',
            )
        ]
    ),
    val=dict(
        type='CUBFSCILDataset',
        data_prefix='./data/CUB_200_2011',
        pipeline=test_pipeline,
        num_cls=100,
        subset='test',
    ),
    test=dict(
        type='CUBFSCILDataset',
        data_prefix='./data/CUB_200_2011',
        pipeline=test_pipeline,
        num_cls=200,
        subset='test',
    ))

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