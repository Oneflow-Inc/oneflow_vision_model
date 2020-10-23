# DCGAN                                                                                                                          
 
## Introduction
DCGAN is an unconditional image generation method, our code is inspired by TensorFlow Tutorial
(https://tensorflow.google.cn/tutorials/generative/dcgan).
 
## Installation
oneflow==0.1.10<br>

 
## Get started
 
### Prepare data
If you want to run train and test on small dataset (e.g., Mnist, about 11M), this project will automatically download this dataset in "./data/mnist".<br>

### Train a model
python of_dcgan.py

### Download pretrained model
If you just want to run test, you can download pretrained model
[here](https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/dcgan.zip), unzip and put it in "./models".
 
### Test a model
python of_dcgan.py --test

