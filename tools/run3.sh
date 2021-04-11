PORT=29004 CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/_my/fpn_r50_512x512_40k_voc12aug1.py 1 --deterministic
PORT=29004 CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/_my/fpn_r50_512x512_40k_voc12aug2.py 1 --deterministic
