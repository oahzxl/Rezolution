_base_ = [
    '../_base_/models/ccnet_r50-d8.py',
    '../_base_/datasets/pascal_voc12.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
