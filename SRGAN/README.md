# SRGAN                                                                                                                       
 
## Introduction
SRGAN is an image super-resolution method, our code is inspired the paper (https://arxiv.org/pdf/1609.04802.pdf).
 
## Installation
oneflow==0.1.10<br>

 
## Get started
 
### Prepare data
Download VOC2012 dataset [here](https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/VOC2012.zip) (about 1.8G), unzip and put it in "./data". Preprocess images to generate *.npy . <br>
python of_data_utils <br>

If you do not want to preprocess images, you can directly download *.npy [here](https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/VOC2012_npy.zip), unzip and put it in "./data". <br>


### Train a model
To use the perceptual loss, you can download perceptual model(vgg16bn) [here](https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/of_vgg16bn_reuse.zip), unzip and put it in "./models".
python of_srgan.py

### Download pretrained model
If you just want to run test, you can download pretrained model [here](https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/srgan.zip), unzip and put it in "./models".
 
### Test a model
You can test an image or images with the same or different shape.
python of_srgan.py --test

