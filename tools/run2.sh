PORT=29009 CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh configs/_my/revnet_psp_80k_down_voc121.py 2
PORT=29009 CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh configs/_my/revnet_psp_80k_fix_voc12.py 2
