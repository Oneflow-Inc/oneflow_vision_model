# TSN

## Introduction
TSN is an action recognition method, our code is inspired by MMAction(https://github.com/open-mmlab/mmaction)

## Installation
oneflow
mmcv

## Get started

### Data preparation
down data,unzip and put it in "./data".
If you want to run train and test on small dataset(about 4.7G), please download dataset here(https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/kinetics400.zip), use kinetics400_train_list_videos_small.txt and kinetics400_val_list_videos_small.txt.
If you want to run train and test on large dataset(about 135G), please download dataset here(https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics), use kinetics400_train_list_videos.txt and kinetics400_val_list_videos.txt.
### download pretrain backbone model
download pretrain backbone model
here(https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/model_pcb.zip),unzip and put it in "./modelzoo"

### Train a model
python tools/train_recognizer_of.py

### Test a model
python tools/test_recognizer_of.py

