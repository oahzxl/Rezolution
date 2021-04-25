_base_ = [
    '../_base_/models/rev_pspnet_r50-d8.py', '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,)
model = dict(
    decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))
