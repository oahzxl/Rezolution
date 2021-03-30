_base_ = [
    '../_base_/models/fpn_red50_neck.py', '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)