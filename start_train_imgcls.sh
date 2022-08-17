#!/bin/bash

opt=$1
gpu=0
basename=`basename $opt`
expid=$(echo $basename | awk  '{ string=substr($0,1,3); print string; }')

echo "Started task, exp ${expid} on GPU no. ${gpu}"
echo $basename

### Check if a directory does not exist ###
if [ ! -d "./logs" ] 
then
    echo "Directory ./logs NOT exists, create now"
    mkdir ./logs
fi

CUDA_VISIBLE_DEVICES=$gpu nohup python -u train_sareo.py --opt $opt > logs/train_${expid}_gpu${gpu}.log 2>&1 &
