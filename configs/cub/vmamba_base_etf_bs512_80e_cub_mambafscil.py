_base_ = [
    '../_base_/models/vmamba_etf.py', '../_base_/datasets/cub_fscil.py',
    '../_base_/schedules/cub_80e.py', '../_base_/default_runtime.py'
]

# CUB requires different inc settings
inc_start = 100
inc_end = 200
inc_step = 10

model = dict(backbone=dict(_delete_=True,
                           type='VMambaBackbone',
                           model_name='vmamba_tiny_s2l5',  # VMamba Tiny model name for MambaNeck
                           pretrained_path='./vssm_tiny_0230_ckpt_epoch_262.pth',  # VMamba Tiny s2l5 pretrained weights
                           out_indices=(0, 1, 2, 3),  # Extract features from all 4 stages
                           frozen_stages=0,  # Freeze patch embedding and first stage
                           channel_first=True),
             neck=dict(type='MambaNeck',
                       version='ss2d',
                       in_channels=768,  # VMamba base stage4 output channels
                       out_channels=768,
                       feat_size=7,  # 224 / (4*8) = 7 (patch_size=4, 4 downsample stages with 2x each)
                       num_layers=3,
                       use_residual_proj=True,
                       # Enhanced skip connection settings (MASC-M) for VMamba features
                       use_multi_scale_skip=True,
                       multi_scale_channels=[96, 192, 384]),
             head=dict(type='ETFHead',
                       in_channels=768,
                       num_classes=200,
                       eval_classes=100,
                       with_len=False,
                       cal_acc=True,
                       loss=dict(type='CombinedLoss', dr_weight=0.0, ce_weight=1.0)),
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

# 증강 이미지용 파이프라인
augmented_pipeline1 = [
    dict(type='LoadAugmentedImage',
         aug_dir='data/CUB_200_2011/augmented_images_1'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]
# 증강 이미지용 파이프라인
augmented_pipeline2= [
    dict(type='LoadAugmentedImage',
         aug_dir='data/CUB_200_2011/augmented_images_2'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]
# 증강 이미지용 파이프라인
augmented_pipeline3 = [
    dict(type='LoadAugmentedImage',
         aug_dir='data/CUB_200_2011/augmented_images_3'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]
# 증강 이미지용 파이프라인
augmented_pipeline4 = [
    dict(type='LoadAugmentedImage',
         aug_dir='data/CUB_200_2011/augmented_images_4'),
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

find_unused_parameters=True