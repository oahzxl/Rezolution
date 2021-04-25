PORT=29006 CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh configs/_my/revnet_psp_40k_up_cat_voc12aug.py 2 --deterministic
PORT=29006 CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh configs/_my/revnet_psp_40k_up_voc12aug.py 2 --deterministic
