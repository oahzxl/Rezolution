PORT=29002 CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/_my/ccnet_r50-d8_512x512_20k_voc12aug1.py 1 --deterministic
PORT=29002 CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/_my/ccnet_r50-d8_512x512_20k_voc12aug2.py 1 --deterministic
PORT=29002 CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/_my/ccnet_r50-d8_512x512_20k_voc12aug3.py 1 --deterministic
