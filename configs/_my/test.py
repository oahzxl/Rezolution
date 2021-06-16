_base_ = [
    '../_base_/models/rev_ocrnet_hr18.py', '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

# optimizer = dict(_delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001)
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))

log_config = dict(interval=20,)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,)
evaluation = dict(interval=100, metric='mIoU')
# model = dict(
#     decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(decode_head=[
    dict(
        type='FCNHead',
        in_channels=[18, 36, 72, 144],
        channels=sum([18, 36, 72, 144]),
        in_index=(0, 1, 2, 3),
        input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    dict(
        type='RevOCRHead',
        in_channels=[18, 36, 72, 144],
        in_index=(0, 1, 2, 3),
        input_transform='resize_concat',
        channels=512,
        mid_channels=128,
        ocr_channels=256,
        dropout_ratio=-1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
])
