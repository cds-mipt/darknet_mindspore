# !/bin/bash

SOURCE=${1:-$(pwd)}
DATASETS=${2:-"/media/serlini/data/Datasets/"}

echo $(echo "Hello $(whoami)")

docker run --rm -it --gpus all \
        -v /dev/shm:/dev/shm \
        -v $SOURCE:/home/darknet/:rw \
        -v $DATASETS:/datasets/:ro \
	-p 8080:8080 \
	-u $(id -u):$(id -g) \
        --name mindspore \
        --privileged=true \
        mindspore/mindspore-gpu:1.3.0
