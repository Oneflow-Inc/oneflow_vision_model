# HrNet                                                                                                                

## Introduction
Ghostnet is a backbone network, our code is inspired the paper (https://arxiv.org/pdf/1902.09212).

## Installation
oneflow==0.3.2<br>


## Get started

### Prepare data
- Please put the coco dataset 2017, including the training set and the test set, into the relevant data set according to the path of `.\dataset\COCO\2017`
- Execute `python write_coco_to_txt.py`, customize the label file of the coco data set.

### Train a model

```
 python3 tranin.py 
```

### Download pretrained model

  The pre-trained model can be downloaded from [BaiduNetdis](https://pan.baidu.com/s/1DrD2TLP3Rv-S85qj3P3xzg) code:w1am.

```
  rm -rf core.* 
  MODEL_LOAD_DIR="hrnet_best_model"
  python3 val.py 
    
```

|         | Train with val(PCK) |
| :-----: | :-----------------: |
| Pytorch |        0.704        |
| Oneflow |        0.696        |

