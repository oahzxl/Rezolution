_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(decode_head=dict(num_classes=21))
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,)

# 0.001效果还是不大行 试试看更小的
