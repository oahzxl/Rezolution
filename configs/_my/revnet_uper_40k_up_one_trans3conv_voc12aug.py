_base_ = [
    '../_base_/models/rev_upernet_r50.py', '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,)
model = dict(
    decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))

# 还是原来的max方法获得卷积核，在psp里用了两个rev，第二个rev的卷积核设成了5、步长为2
