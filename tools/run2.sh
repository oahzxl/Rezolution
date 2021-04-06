CUDA_VISIBLE_DEVICES=1 bash tools/dist_train.sh configs/_my/my3.py 1 --deterministic
sleep 1200s
CUDA_VISIBLE_DEVICES=1 bash tools/dist_train.sh configs/_my/my4.py 1 --deterministic
