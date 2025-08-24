# model settings
model = dict(type='ImageClassifierCIL',
             backbone=dict(type='ResNet12', with_avgpool=False, flatten=False, out_indices=(0, 1, 2, 3)),
             neck=dict(type='MambaNeck', in_channels=640, out_channels=512),
             head=dict(
                 type='ETFHead',
                 num_classes=100,
                 eval_classes=60,
                 in_channels=512,
                 topk=(1, 5),
                 cal_acc=True,
             ))
