_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/pascal_voc12_256.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,)
