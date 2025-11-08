_base_ = [
    '../_base_/models/vmamba_etf.py',
    '../_base_/datasets/mini_imagenet_fscil.py',
    '../_base_/schedules/mini_imagenet_5e.py', '../_base_/default_runtime.py'
]

model = dict(backbone=dict(_delete_=True,
                           type='VisionTransformer',
                           arch='base',
                           img_size=168,
                           patch_size=16,
                           out_indices=-1,
                           final_norm=True,
                           with_cls_token=True,
                           output_cls_token=False,
                      init_cfg=dict(
                          type='Pretrained',
                          checkpoint='https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth',
                          prefix='backbone.')),
             neck=dict(type='MoEFSCILNeck',
                      num_experts=4,
                      top_k=4,
                      eval_top_k=4,
                      in_channels=768,
                      out_channels=768,
                      feat_size=11,
                      use_aux_loss=False,
                      aux_loss_weight=0.01),
             head=dict(type='ETFHead',
                       in_channels=768,
                       loss=dict(type='CombinedLoss', dr_weight=1.0, ce_weight=0.0),
                       with_len=True),
             mixup=0.5,
             mixup_prob=0.3)

base_copy_list = (1, 2, 3, 4, 5, 6, 7, 8, None, None)
step_list = (100, 110, 120, 130, 140, 150, 160, 170, None, None)
copy_list = (10, 10, 10, 10, 10, 10, 10, 10, None, None)

finetune_lr = 0.1

# optimizer
optimizer = dict(type='SGD',
                 lr=finetune_lr,
                 momentum=0.9,
                 weight_decay=0.0005,
                 paramwise_cfg=dict(
                     custom_keys={
                         'neck.pos_embed': dict(lr_mult=1.0),
                         'neck.moe.gate': dict(lr_mult=1.0),     
                         'neck.moe.experts': dict(lr_mult=1.0),  
                     }))

find_unused_parameters = True