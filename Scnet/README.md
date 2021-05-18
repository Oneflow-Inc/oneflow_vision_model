# Scnet                                                                                                                

## Introduction
Scnet is a vehicle color classification network based on scloss, which can be easily combined with different backbone networks.

## Installation
oneflow==0.3.4 <br>
visdom==0.1.8.9<br>
pillow==8.2.0
numpy==1.17.4
## Get started
### Prepare data
<br>Data preparation<br>
If you want to run train and test on  dataset, please download dataset [here](http://cloud.eic.hust.edu.cn:8071/~pchen/color.rar).
<br>How to use the dataset<br>
If you use the vehicle color recognition dataset for testing your recognition algorithm you should try and make your results comparable to the results of others. 
We suggest to choose half of the images in each category to train a model. 
And use the other half images to test the recognition algorithm.

### Train a model
We have installed visdom to visualize the training model, 
and run the following program to enter http://localhost:8097/ get the training curve.
```
python -m visdom.server
```
Train with the following command.
```
 python train.py 

```

### Download pretrained model

The pre-trained model can be downloaded from [BaiduNetdis](https://pan.baidu.com/s/1P859zYflN1yUIPIfzkK3jg).
Code fbc4

### Test model
 ```
 python inference.py --model_load_dir ./model/scnet_hustcolor_90.2_top1
 ```

### Performer of model
|         | val(Top1) |
| :-----: | :-----------------: |
| resnet  |        0.892        |
| scnet   |        0.907        |

