# VMamba backbone + ETF head configuration for FSCIL
model = dict(
    type='ImageClassifierCIL',
    backbone=dict(
        type='VMambaBackbone',
        model_name='vmamba_tiny_s1l8',  # VMamba Tiny model name for MambaNeck
        pretrained_path='./vssm1_tiny_0230s_ckpt_epoch_264.pth',  # VMamba Tiny pretrained weights
        out_indices=(0, 1, 2, 3),  # Multi-scale feature extraction
        frozen_stages=1,  # Freeze first stage to prevent overfitting
        channel_first=True,
    ),
    neck=dict(
        type='MambaNeck',
        version='ss2d',
        in_channels=1024,  # VMamba base final layer channels
        out_channels=1024,
        feat_size=7,  # 224/32 = 7 (after 4 downsample stages)
        num_layers=2,
        use_residual_proj=True,
        # Enhanced skip connection settings for VMamba multi-scale features
        use_multi_scale_skip=True,
        multi_scale_channels=[128, 256, 512],  # VMamba base: [128, 256, 512, 1024]
        d_state=256,
        ssm_expand_ratio=1.0,
    ),
    head=dict(
        type='ETFHead',
        in_channels=1024,
        num_classes=200,
        eval_classes=100,
        with_len=False,
        cal_acc=True
    ),
    mixup=0,
    mixup_prob=0
) 
