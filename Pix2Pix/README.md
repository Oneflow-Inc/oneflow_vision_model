# Pix2Pix                                                                                                                       
 
## Introduction
Pix2Pix is a conditional image generation method, our code is inspired by TensorFlow Tutorial
(https://tensorflow.google.cn/tutorials/generative/pix2pix).
 
## Installation
oneflow==0.1.10<br>

 
## Get started
 
### Prepare data
If you want to run train and test on small dataset (e.g., Facades, about 30.9M), this project will automatically download this dataset in "./data/facades".<br>

### Train a model
python of_pix2pix.py

### Download pretrained model
If you just want to run test, you can download pretrained model
[here](https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/pix2pix.zip), unzip and put it in "./models".
 
### Test a model
python of_pixpix.py --test

