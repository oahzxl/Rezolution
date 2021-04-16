PORT=29009 CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/_my/deeplabv3plus_r50-d8_512x512_40k_voc12aug1.py 1
PORT=29009 CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/_my/deeplabv3plus_r50-d8_512x512_40k_voc12aug2.py 1
PORT=29009 CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/_my/deeplabv3plus_r50-d8_512x512_40k_voc12aug3.py 1
