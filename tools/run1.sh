PORT=29006 CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh configs/_my/revnet_psp_40k_up_voc12aug1.py 2 --deterministic
PORT=29006 CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh configs/_my/revnet_psp_40k_up_voc12aug2.py 2 --deterministic
