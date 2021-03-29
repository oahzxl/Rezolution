bash tools/dist_train.sh configs/_my/fpn_red50_neck_512x1024_40k_cityscapes1.py 2
bash tools/dist_train.sh configs/_my/fpn_red50_neck_512x1024_40k_cityscapes2.py 2
bash tools/dist_train.sh configs/_my/fpn_red50_neck_512x1024_40k_cityscapes3.py 2
bash tools/dist_train.sh configs/_my/ccnet_r50-d8_512x512_40k_voc12.py 2 --deterministic
