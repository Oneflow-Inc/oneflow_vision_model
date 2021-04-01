# TTSR                                                                                                                       
 
## Introduction
TTSR is an image super-resolution method, our code is inspired the paper (https://arxiv.org/pdf/2006.04139.pdf).
 
## Installation
oneflow==0.3.2<br>

 
## Get started
 
### Prepare data
Download [CUFED train set](https://drive.google.com/drive/folders/1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I) and [CUFED test set](https://drive.google.com/file/d/1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph/view), unzip and put it in "./data".

### Train a model
To use the perceptual loss, you can download perceptual model(vgg16bn) [here](https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/of_vgg16bn_reuse.zip), unzip and put it in "./models".
python of_ttsr.py

### Download pretrained model
If you just want to run test, you can download pretrained model [here](https://drive.google.com/file/d/1puZCYIlehZHIhpOQBsCGBc7SprB8ioik/view?usp=sharing), unzip and put it in "./models".
 
### Test a model
python of_ttsr.py --test

### Compare
|         |     PSNR      |      SSIM    |
| :------ | :-----------: | -----------: |
| PyTorch |    25.53      |     0.765    |
| OneFlow |    25.30      |     0.764


