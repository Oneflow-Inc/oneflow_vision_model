# MobilenetV3

## Introduction

MobileNetV3 is a light network aiming at object classfication. It can also been used as backbone of other computer vision networks, e.g., instance segmentation, depth estimation.

## Installation

oneflow==0.3.5

## Get started

### Prepare data
Refer to the oneflow using cnn-classification to prepare the corresponding data set.

### Train a model

```
python3 of_cnn_train_val.py \
  --model="mobilenetv3"
```

### Compare
