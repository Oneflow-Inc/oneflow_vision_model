# MobilenetV3

## Introduction

MobileNetV3 is a light network aiming at object classfication. It can also been used as backbone of other computer vision networks, e.g., instance segmentation, depth estimation.

MobileNetV3 contains two models, i.e., MobileNetV3-Large and MobileNetV3-Small. Both of the models are written in the python file and can be applied.

## Installation

oneflow==0.3.5

## Get started

### Prepare data

Refer to the oneflow using [cnn-classification](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns) to prepare the corresponding data set.

### Train a model 

Before training, it is necessary to add mobilenetv3_large and mobilenetv3_small model by changing part of_cnn_train_val.py:

```
model_dict = {
    "resnet50": resnet_model.resnet50,
    "vgg": vgg_model.vgg16bn,
    "alexnet": alexnet_model.alexnet,
    "inceptionv3": inception_model.inceptionv3,
    "mobilenetv2": mobilenet_v2_model.Mobilenet,
    "mobilenetv3_large": mobilenet_v3_model.Mobilenet_Large,
    "mobilenetv3_small": mobilenet_v3_model.Mobilenet_Small,
    "resnext50": resnext_model.resnext50,
}
```
Also, import mobilenet_v3_model should be written in of_cnn_train_val.py.

Then write the following command to train the model. Pay attention that either mobilenetv3_large or mobilenetv3_small should be defined.

```
python3 of_cnn_train_val.py \
  --model="mobilenetv3_large"
```

It is also practical to change config.py directly to train the model with mobilenetv3_large or mobilenetv3_small.

### Compare
