#!/bin/bash
wget -q $1 -O test.jpg
sudo docker run -a stdout --volume=$(pwd):/workspace 243284212115.dkr.ecr.us-east-1.amazonaws.com/whi_nsfw:15FEB2017 \
python ./classify_nsfw.py \
--model_def nsfw_model/deploy.prototxt \
--pretrained_model nsfw_model/resnet_50_1by2_nsfw.caffemodel \
test.jpg
rm test.jpg
