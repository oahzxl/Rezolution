CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/_my/my1.py 1 --deterministic
sleep 1200s
CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/_my/my2.py 1 --deterministic
