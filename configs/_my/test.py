_base_ = [
    '../_base_/models/revnet_fpn_neck.py', '../_base_/datasets/pascal_voc12_256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

# optimizer = dict(_delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001)
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))

log_config = dict(interval=50,)
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,)
evaluation = dict(interval=200, metric='mIoU')
