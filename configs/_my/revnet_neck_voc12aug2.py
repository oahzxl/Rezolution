_base_ = [
    '../_base_/models/revnet_fpn_neck.py', '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,)
model = dict(decode_head=dict(num_classes=21))
