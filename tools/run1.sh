PORT=29008 CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/_my/deeplabv3_r50-d8_512x512_40k_voc12aug1.py 1 --deterministic
sleep 5min
PORT=29008 CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/_my/fpn_r50_512x512_40k_voc12aug1.py 1 --deterministic
sleep 5min
PORT=29008 CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/_my/deeplabv3_r50-d8_512x512_40k_voc12aug2.py 1 --deterministic
sleep 5min
PORT=29008 CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/_my/fpn_r50_512x512_40k_voc12aug2.py 1 --deterministic
sleep 5min
PORT=29008 CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/_my/deeplabv3_r50-d8_512x512_40k_voc12aug3.py 1 --deterministic
sleep 5min
PORT=29008 CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/_my/fpn_r50_512x512_40k_voc12aug3.py 1 --deterministic
sleep 5min
PORT=29008 CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/_my/fpn_r50_512x512_40k_voc12aug4.py 1 --deterministic