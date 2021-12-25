CUDA_VISIBLE_DEVICES=2,0 PORT=29004 bash tools/dist_train.sh configs/_my/rev_ca_attnx2_ocrnet_hr18_512x512_40k_voc12aug.py 2 --seed 0 --deterministic
CUDA_VISIBLE_DEVICES=2,0 PORT=29004 bash tools/dist_train.sh configs/_my/rev_ca_attnx2_ocrnet_hr18_512x512_40k_voc12aug.py 2 --seed 1000 --deterministic
CUDA_VISIBLE_DEVICES=2,0 PORT=29004 bash tools/dist_train.sh configs/_my/rev_ca_attnx2_ocrnet_hr18_512x512_40k_voc12aug.py 2 --seed 2000 --deterministic
