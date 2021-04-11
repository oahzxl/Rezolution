PORT=29003 CUDA_VISIBLE_DEVICES=1 bash tools/dist_train.sh configs/_my/pspnet_r50-d8_512x512_20k_voc12aug1.py 1 --deterministic
PORT=29003 CUDA_VISIBLE_DEVICES=1 bash tools/dist_train.sh configs/_my/pspnet_r50-d8_512x512_20k_voc12aug2.py 1 --deterministic
PORT=29003 CUDA_VISIBLE_DEVICES=1 bash tools/dist_train.sh configs/_my/pspnet_r50-d8_512x512_20k_voc12aug3.py 1 --deterministic
