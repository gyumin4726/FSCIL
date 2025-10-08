# VMamba backbone + ETF head configuration for FSCIL
model = dict(
    type='ImageClassifierCIL',
    backbone=dict(
        type='VMambaBackbone',
        model_name='vmamba_base_s2l15', 
        pretrained_path='./vssm_base_0229_ckpt_epoch_237.pth',
        out_indices=(0, 1, 2, 3),  
        frozen_stages=0,
        channel_first=True,
    ),
    neck=dict(
        type='MambaNeck',
        in_channels=1024,
        out_channels=1024,
        feat_size=7,
    ),
    head=dict(
        type='ETFHead',
        in_channels=1024,
        num_classes=100,
        eval_classes=60,
        with_len=False,
        cal_acc=True
    ),
    mixup=0,
    mixup_prob=0
) 
