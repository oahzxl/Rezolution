CUDA_VISIBLE_DEVICES=2,3 PORT=29002 bash tools/dist_train.sh configs/_my/rev_ca_mid32_ocrnet_hr18_512x512_40k_voc12aug.py 2 --deterministic
CUDA_VISIBLE_DEVICES=2,3 PORT=29002 bash tools/dist_train.sh configs/_my/rev_ca_mid128_ocrnet_hr18_512x512_40k_voc12aug.py 2 --deterministic
