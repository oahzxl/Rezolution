CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/_my/deeplabv3plus_r50-d8_512x512_40k_voc12.py 1 --deterministic
sleep 300s
CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/_my/fpn_red50_neck_512x512_20k_voc12.py 1
sleep 300s
CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/_my/ccnet_r50-d8_512x512_40k_voc12.py 1 --deterministic
