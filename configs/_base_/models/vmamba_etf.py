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
        in_channels=1024,  # VMamba base final layer channels
        out_channels=1024,
        feat_size=7,
        num_layers=2,
        use_residual_proj=True,
        use_multi_scale_skip=False,
        multi_scale_channels=[128, 256, 512],  # VMamba base: [128, 256, 512, 1024]
        d_state=256,
        ssm_expand_ratio=1.0,
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
