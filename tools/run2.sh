CUDA_VISIBLE_DEVICES=1,2 PORT=29002 bash tools/dist_train.sh configs/_my/ocrnet_hr18_512x512_40k_voc12aug.py 2 --seed 0 --deterministic
CUDA_VISIBLE_DEVICES=1,2 PORT=29002 bash tools/dist_train.sh configs/_my/ocrnet_hr18_512x512_40k_voc12aug.py 2 --seed 1000 --deterministic
