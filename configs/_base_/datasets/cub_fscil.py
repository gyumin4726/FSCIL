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

# # 증강 이미지용 파이프라인
# augmented_pipeline = [
#     dict(type='LoadAugmentedImage',
#          aug_dir='data/CUB_200_2011/augmented_images'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
# ]

# # 증강 이미지용 파이프라인
# augmented_pipeline1 = [
#     dict(type='LoadAugmentedImage',
#          aug_dir='data/CUB_200_2011/augmented_images_1'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
# ]
# # 증강 이미지용 파이프라인
# augmented_pipeline2= [
#     dict(type='LoadAugmentedImage',
#          aug_dir='data/CUB_200_2011/augmented_images_2'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
# ]
# # 증강 이미지용 파이프라인
# augmented_pipeline3 = [
#     dict(type='LoadAugmentedImage',
#          aug_dir='data/CUB_200_2011/augmented_images_3'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
# ]
# # 증강 이미지용 파이프라인
# augmented_pipeline4 = [
#     dict(type='LoadAugmentedImage',
#          aug_dir='data/CUB_200_2011/augmented_images_4'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
# ]
# # 회전된 이미지용 파이프라인 (별도 폴더)
# rotated_pipeline = [
#     dict(type='LoadRotatedImage', rot_dir='data/CUB_200_2011/rotated_images'),
#     dict(type='Resize', size=(_img_resize_size, _img_resize_size)),
#     dict(type='CenterCrop', crop_size=img_size),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
# ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(_img_resize_size, -1)),
    dict(type='CenterCrop', crop_size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train_dataloader=dict(persistent_workers=True),
    val_dataloader=dict(persistent_workers=True),
    test_dataloader=dict(persistent_workers=True),
    train=dict(type='RepeatDataset',
                times=1,
                dataset=dict(
                    type='CUBFSCILDataset',
                    data_prefix='./data/CUB_200_2011',
                    pipeline=original_pipeline,
                    num_cls=100,
                    subset='train',
                )),
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