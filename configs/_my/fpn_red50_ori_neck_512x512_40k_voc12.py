_base_ = [
    '../_base_/models/fpn_red50_neck_ori.py', '../_base_/datasets/pascal_voc12_256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,)
model = dict(decode_head=dict(num_classes=21))
