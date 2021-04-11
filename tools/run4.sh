PORT=29004 CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/_my/revnet_neck_voc12aug4.py 1 --deterministic
PORT=29004 CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/_my/revnet_neck_voc12aug5.py 1 --deterministic
PORT=29004 CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/_my/revnet_neck_voc12aug6.py 1 --deterministic
PORT=29004 CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/_my/revnet_neck_voc12aug7.py 1 --deterministic