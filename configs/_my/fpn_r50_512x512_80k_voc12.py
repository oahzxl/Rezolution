_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/pascal_voc12_256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(decode_head=dict(num_classes=21))
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,)
