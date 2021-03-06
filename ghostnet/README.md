# Ghostnet                                                                                                                       
 
## Introduction
Ghostnet is a backbone network, our code is inspired the paper (https://arxiv.org/abs/1911.11907).
 
## Installation
oneflow==0.3.2<br>


## Get started
 
### Prepare data
Refer to the oneflow using [cnn-classification](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns) to prepare the corresponding data set.

### Train a model

```
 python3 of_cnn_train_val.py \
    --train_data_dir=$DATA_ROOT/train \
    --train_data_part_num=256 \
    --val_data_dir=$DATA_ROOT/validation \
    --val_data_part_num=256 \
    --num_nodes=1 \
    --gpu_num_per_node=8 \
    --optimizer="rmsprop"  \
    --decay_rate=0.9 \
    --momentum=0.9 \
    --learning_rate=0.4 \
    --wd=0.00004 \
    --lr_decay="exponential" \
    --lr_decay_rate=0.94 \
    --lr_decay_epochs=2 \
    --loss_print_every_n_iter=100 \
    --batch_size_per_device=128 \
    --val_batch_size_per_device=128 \
    --num_epoch=800 \
    --warmup_epochs=0 \
    --model="ghostnet" \
```

### Download pretrained model

  The pre-trained model can be downloaded from [BaiduNetdis](https://pan.baidu.com/s/1J6wfRS3AT9r7VenqHf9PDA) code:64bp .

```
  rm -rf core.* 
  MODEL_LOAD_DIR="ghostnet_best_model"
  python3 of_cnn_inference.py \
    --model="ghostnet" \
    --image_path="data/fish.jpg" \
    --model_load_dir=$MODEL_LOAD_DIR
```

### Compare

|         | Validation TOP1 (%) | Validation TOPK (%) |
| :------ | :----------------:  | ------------------: |
|  Paper  |         66.2        |         86.6        |
| OneFlow |         66.8        |       87.5 
