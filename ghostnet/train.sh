#!/bin/bash
rm -rf core.* 
rm -rf ./output/ghostnet/snapshots/*

#  # training with synthetic data
#  python3 of_cnn_train_val.py \
#      --num_examples=50 \
#      --num_val_examples=50 \
#      --num_nodes=1 \
#      --gpu_num_per_node=1 \
#      --model_update="momentum" \
#      --learning_rate=0.001 \
#      --loss_print_every_n_iter=1 \
#     --batch_size_per_device=16 \
#      --val_batch_size_per_device=16 \
#      --num_epoch=10 \
#      --model="resnet50"



# # training with mini-imagenet
# DATA_ROOT=data/mini-imagenet/ofrecord
# python3 of_cnn_train_val.py \
#     --train_data_dir=$DATA_ROOT/train \
#     --num_examples=50 \
#     --train_data_part_num=1 \
#     --val_data_dir=$DATA_ROOT/validation \
#     --num_val_examples=50 \
#     --val_data_part_num=1 \
#     --num_nodes=1 \
#     --gpu_num_per_node=1 \
#     --model_update="momentum" \
#     --learning_rate=0.001 \
#     --loss_print_every_n_iter=1 \
#     --batch_size_per_device=16 \
#     --val_batch_size_per_device=10 \
#     --num_epoch=10 \
#     --model="resnet50"




# training with imagenet
DATA_ROOT=data/imagenet/ofrecord
LOG_FOLDER=../logs
mkdir -p $LOG_FOLDER
LOGFILE=$LOG_FOLDER/ghostnet_training.log

python3 of_cnn_train_val.py \
    --train_data_dir=$DATA_ROOT/train \
    --train_data_part_num=256 \
    --val_data_dir=$DATA_ROOT/validation \
    --val_data_part_num=256 \
    --num_nodes=1 \
    --gpu_num_per_node=8 \
    --model_update="momentum" \
    --learning_rate=0.256 \
    --loss_print_every_n_iter=100 \
    --batch_size_per_device=64 \
    --val_batch_size_per_device=50 \
    --num_epoch=90 \
    --model="ghostnet" 2>&1 | tee ${LOGFILE}

echo "Writting log to ${LOGFILE}" 

