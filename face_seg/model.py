#-*- coding:utf-8 -*-
"""
# Version:0.0.1
# Date:08/07/2020
# Author: Riling Wei (weirl@zhejianglab.com)
"""

import oneflow as flow
import argparse
import cv2
import numpy as np
import os
import time


func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

# set up initialize i
def _get_kernel_initializer():
    return flow.variance_scaling_initializer(distribution="random_normal", data_format="NCHW")

def _get_regularizer():
    return flow.regularizers.l2(0.0005)

def _get_bias_initializer():
    return flow.zeros_initializer()

# upsample convolution layer
def _conv2d_transpose_layer(
    name,  # name of layer
    input, # input of layer
    kernel_size,  # kernel size of filters
    strides=1,    # strides size
    padding="SAME",     # padding is SAME or VALID
    data_format="NCHW", # N:batch size C: Number of channels H:height W:width
    dilations=1,
    trainable=False,   # trainable is True or False
    input_shape = None,
    output_shape = None,
    use_bias=True,     # use_bias is True or False
    bias_initializer= _get_bias_initializer() #flow.random_uniform_initializer(),
):
    in_channels = input_shape[1]
    out_channels = output_shape[1]
    dilations = 1

    # weights in convolution layers
    weight = flow.get_variable(
        name+"-weight",
        shape=(in_channels, out_channels, kernel_size, kernel_size),
        dtype=flow.float,
        initializer= _get_kernel_initializer(),# flow.random_uniform_initializer(minval=-10, maxval=10), # initialise weight
        regularizer= _get_regularizer(), # weight regularizer
        trainable=False,
    )
    output = flow.nn.conv2d_transpose(input, weight, strides=strides, output_shape=output_shape, dilations=dilations,
                                       padding=padding, data_format=data_format) # deconvolution layer
    # bias in convolution layers
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(out_channels,),
            dtype=input.dtype,
            initializer= _get_bias_initializer(), # bias_initializer, # initialise bias
            regularizer= _get_regularizer() # bias regularizer
        )
        # add bias if use_bias is true
        output = flow.nn.bias_add(output, bias, data_format)

    return output
# convolution layer
def _conv2d_layer(
    name, # name of layer
    input, # input of layer
    filters,
    kernel_size, # kernel size of filters
    strides=1, # strides size
    padding="SAME", # padding is SAME or VALID
    data_format="NCHW", # N:batch size C: Number of channels H:height W:width
    dilations=1,
    trainable=True, # trainable is True or False
    use_bias=True, # use_bias is True or False
    weight_initializer= _get_kernel_initializer(), #flow.variance_scaling_initializer(data_format="NCHW"),
    bias_initializer= _get_bias_initializer() # flow.random_uniform_initializer(),
):
    # weights in deconvolution layers
    weight = flow.get_variable(
        name + "-weight",
        shape=(filters, input.shape[1], kernel_size, kernel_size),
        dtype=input.dtype,
        initializer=weight_initializer, # initialise weight
        trainable=trainable,
    )

    output = flow.nn.conv2d(input, weight, strides, padding, data_format, dilations, name=name) # deconvolution layer
    # bias in deconvolution layers
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer= _get_bias_initializer() # bias_initializer, # initialise bias
        )
        # add bias if use_bias is true
        output = flow.nn.bias_add(output, bias, data_format)
    return output

# structure of decoder
def DecoderBlockLinkNet(input,in_channels, n_filters,stage_name,d_output_shape):
    # convolution layer, number of filters: in_channels//4
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=in_channels//4, kernel_size=1, strides=1, padding="SAME"
    )

    batchNormal1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"norm1")
    # activatation
    relu1 = flow.math.relu(batchNormal1,name=stage_name+"reul1")
    # set output shape
    output_shape = d_output_shape
    # unsample layer
    deconv2 = _conv2d_transpose_layer(name=stage_name+'deconv2',input=relu1,input_shape=relu1.shape,output_shape=output_shape,kernel_size=4,strides=2)

    batchNormal2 = flow.layers.batch_normalization(deconv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"norm2")
    # activatation
    relu2 = flow.math.relu(batchNormal2,name=stage_name+"reul2")
    # convolution layer, number of filters: n_filters
    conv3 = _conv2d_layer(
        stage_name+"conv3", relu2, filters=n_filters, kernel_size=1, strides=1, padding="SAME"
    )

    batchNormal3 = flow.layers.batch_normalization(conv3, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"norm3") # bn
    # activatation
    relu3 = flow.math.relu(batchNormal3,name=stage_name+"reul3")
    return relu3

# structure in layer1
def BasicBlock_layer1(input,stage_name):
    # convolution layer, number of filters: 64
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=64, kernel_size=3, strides=1, trainable=False,use_bias=False,
    )

    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")
    # activatation
    relu = flow.math.relu(bn1,name=stage_name+"relu1")
    # convolution layer, number of filters: 64
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=64, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")
    # output = bn2 + residual
    bn2 += input # output = bn2 + residual
    # activatation
    relu2 = flow.math.relu(bn2,name=stage_name+"relu2")
    return relu2

# downsample in basicblock of resnet
def downsample(input,filters,kernel_size,strides,stage_name):
    # convolution layer, number of filters: filters
    conv1 = _conv2d_layer(
        stage_name+"downsample-conv1", input, filters=filters, kernel_size=kernel_size, strides=strides, padding="SAME",use_bias=False
    )
    #bn
    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"downsample-bn1")

    return bn1
# structure of block1 in layer2
def BasicBlock_layer2_0(input,stage_name):
    # convolution layer, number of filters: 128
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=128, kernel_size=3, strides=2,use_bias=False,
    )
    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")
    relu = flow.math.relu(bn1,name=stage_name+"relu1")
    # convolution layer, number of filters: 128
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=128, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")
    down_input = downsample(input,128,1,2,stage_name)
    # with downsample as residual
    # output = bn2 + residual
    bn2 += down_input
    relu2 = flow.math.relu(bn2,name=stage_name+"relu2")
    return relu2
# structure of block2 in layer2
def BasicBlock_layer2_1(input,stage_name):
    #input = output of BasicBlock_layer2_0
    # convolution layer, number of filters: 128
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=128, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )

    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")

    relu = flow.math.relu(bn1,name=stage_name+"relu")
    # convolution layer, number of filters: 128
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=128, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )

    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")
    # output = bn2 + residual
    bn2 += input
    relu2 = flow.math.relu(bn2,name=stage_name+"relu2")
    return relu2
# structure of block3 in layer2
def BasicBlock_layer2_2(input,stage_name):

    # input = output of BasicBlock_layer2_1
    # convolution layer, number of filters: 128
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=128, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")
    relu = flow.math.relu(bn1,name=stage_name+"relu")
    # convolution layer, number of filters: 128
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=128, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")
    # output = bn2 + residual
    bn2 += input
    relu2 = flow.math.relu(bn2,name=stage_name+"relu2")
    return relu2
# structure of block4 in layer2
def BasicBlock_layer2_3(input,stage_name):
    # input = output of BasicBlock_layer2_2
    # convolution layer, number of filters: 128
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=128, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )

    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")

    relu = flow.math.relu(bn1,name=stage_name+"relu")
    # convolution layer, number of filters: 128
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=128, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )

    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")
    # output = bn2 + residual
    bn2 += input
    relu2 = flow.math.relu(bn2, name=stage_name + "relu2")
    return relu2
# structure of block1 in layer3
def BasicBlock_layer3_0(input,stage_name):

    # first block in layer3
    # convolution layer, number of filters: 256
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=256, kernel_size=3, strides=2, padding="SAME",use_bias=False,
    )
    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")
    relu = flow.math.relu(bn1,name=stage_name+"relu")
    # convolution layer, number of filters: 256
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=256, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")

    downsample_input = downsample(input,256,1,2,stage_name)
    # with downsample as residual
    bn2 += downsample_input
    relu2 = flow.math.relu(bn2, name=stage_name + "relu2")
    return relu2
# structure of block2 in layer3
def BasicBlock_layer3_1(input,stage_name):

    # input = output of BasicBlock_layer3_0
    # convolution layer, number of filters: 256
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=256, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")
    relu = flow.math.relu(bn1,name=stage_name+"relu")
    # convolution layer, number of filters: 256
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=256, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")
    # output = bn2 + residual
    bn2 += input
    relu2 = flow.math.relu(bn2, name=stage_name + "relu2")
    return relu2
# structure of block3 in layer3
def BasicBlock_layer3_2(input,stage_name):
    # input = output of BasicBlock_layer3_1
    # convolution layer, number of filters: 256
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=256, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")
    relu = flow.math.relu(bn1,name=stage_name+"relu")
    # convolution layer, number of filters: 256
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=256, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")
    # output = bn2 + residual
    bn2 += input
    relu2 = flow.math.relu(bn2, name=stage_name + "relu2")
    return relu2
# structure of block4 in layer3
def BasicBlock_layer3_3(input,stage_name):
    # input = output of BasicBlock_layer3_2
    # convolution layer, number of filters: 256
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=256, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")
    relu = flow.math.relu(bn1,name=stage_name+"relu")
    # convolution layer, number of filters: 256
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=256, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")
    # output = bn2 + residual
    bn2 += input
    relu2 = flow.math.relu(bn2, name=stage_name + "relu2")
    return relu2
# structure of block5 in layer3
def BasicBlock_layer3_4(input,stage_name):
    # input = output of BasicBlock_layer3_3
    # convolution layer, number of filters: 256
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=256, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")
    relu = flow.math.relu(bn1,name=stage_name+"relu")
    # convolution layer, number of filters: 256
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=256, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")
    # output = bn2 + residual
    bn2 += input
    relu2 = flow.math.relu(bn2, name=stage_name + "relu2")
    return relu2
# structure of block6 in layer3
def BasicBlock_layer3_5(input,stage_name):
    # input = output of BasicBlock_layer3_4
    # convolution layer, number of filters: 256
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=256, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")
    relu = flow.math.relu(bn1,name=stage_name+"relu")
    # convolution layer, number of filters: 256
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=256, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")
    # output = bn2 + residual
    bn2 += input
    relu2 = flow.math.relu(bn2, name=stage_name + "relu2")
    return relu2
# structure of block1 in layer4
def BasicBlock_layer4_0(input,stage_name):

    # first layer of layer4
    # convolution layer, number of filters: 512
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=512, kernel_size=3, strides=2, padding="SAME",use_bias=False,
    )
    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")
    relu = flow.math.relu(bn1,name=stage_name+"relu")
    # convolution layer, number of filters: 256
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=512, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")

    downsample_input = downsample(input,512,1,2,stage_name)
    # with downsample as residual
    # output = bn2 + residual
    bn2 += downsample_input
    relu2 = flow.math.relu(bn2, name=stage_name + "relu2")
    return relu2
# structure of block2 in layer4
def BasicBlock_layer4_1(input,stage_name):
    # input = output of BasicBlock_layer4_0
    # convolution layer, number of filters: 512
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=512, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")
    relu = flow.math.relu(bn1,name=stage_name+"relu")
    # convolution layer, number of filters: 512
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=512, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")
    # output = bn2 + residual
    bn2 += input
    relu2 = flow.math.relu(bn2, name=stage_name + "relu2")
    return relu2
# structure of block3 in layer4
def BasicBlock_layer4_2(input,stage_name):
    # input = output of BasicBlock_layer4_1
    # convolution layer, number of filters: 512
    conv1 = _conv2d_layer(
        stage_name+"conv1", input, filters=512, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn1 = flow.layers.batch_normalization(conv1, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn1")
    relu = flow.math.relu(bn1,name=stage_name+"relu")
    # convolution layer, number of filters: 512
    conv2 = _conv2d_layer(
        stage_name+"conv2", relu, filters=512, kernel_size=3, strides=1, padding="SAME",use_bias=False,
    )
    bn2 = flow.layers.batch_normalization(conv2, axis=1, momentum=0.1, epsilon=0.00001,name=stage_name+"bn2")
    # output = bn2 + residual
    bn2 += input
    relu2 = flow.math.relu(bn2, name=stage_name + "relu2")
    return relu2

# layer1 = resnet34.layer4 in Pytorch
def layer1(input):
    # basic block1
    stage_name = "encoder1-0-"
    block1 = BasicBlock_layer1(input,stage_name)
    # basic block2
    stage_name = "encoder1-1-"
    block2 = BasicBlock_layer1(block1,stage_name)
    # basic block3
    stage_name = "encoder1-2-"
    block3 = BasicBlock_layer1(block2,stage_name)

    return block3

# layer2 = resnet34.layer2 in Pytorch
def layer2(input):
    # input = output from layer1
    # basic block1
    stage_name = "encoder2-0-"
    block1 = BasicBlock_layer2_0(input,stage_name)
    stage_name = "encoder2-1-"
    # basic block2
    block2 = BasicBlock_layer2_1(block1,stage_name)
    stage_name = "encoder2-2-"
    # basic block3
    block3 = BasicBlock_layer2_2(block2,stage_name)
    stage_name = "encoder2-3-"
    # basic block4
    block4 = BasicBlock_layer2_3(block3,stage_name)
    return block4

# layer3 = resnet34.layer3 in Pytorch
def layer3(input):
    # input = output from layer2
    stage_name = "encoder3-0-"
    # basic block1
    block1 = BasicBlock_layer3_0(input,stage_name)
    stage_name = "encoder3-1-"
    # basic block2
    block2 = BasicBlock_layer3_1(block1,stage_name)
    stage_name = "encoder3-2-"
    # basic block3
    block3 = BasicBlock_layer3_2(block2,stage_name)
    stage_name = "encoder3-3-"
    # basic block4
    block4 = BasicBlock_layer3_3(block3,stage_name)
    stage_name = "encoder3-4-"
    # basic block5
    block5 = BasicBlock_layer3_4(block4,stage_name)
    stage_name = "encoder3-5-"
    # basic block6
    block6 = BasicBlock_layer3_5(block5,stage_name)
    return block6

# layer4 = resnet34.layer4 in Pytorch
def layer4(input):
    # input = output from layer3
    stage_name = "encoder4-0-"
    # basic block1
    block1 = BasicBlock_layer4_0(input,stage_name)
    stage_name = "encoder4-1-"
    # basic block2
    block2 = BasicBlock_layer4_1(block1,stage_name)
    stage_name = "encoder4-2-"
    # basic block3
    block3 = BasicBlock_layer4_2(block2,stage_name)
    return block3

def finaldeconv1(input,stage_name,output_shape):
    # final deconv1
    deconv2 = _conv2d_transpose_layer(name=stage_name,input=input,input_shape=input.shape,output_shape=output_shape,kernel_size=3,strides=2,padding='VALID')
    return deconv2

def finalrelu1(input):
    # final relu1
    return flow.math.relu(input,name="finalrelu1")

def finalconv2(input):
    # final conv2
    return _conv2d_layer(
        "finalconv2", input, filters=32, kernel_size=3, strides=1, padding="VALID",
    )
def finalrelu2(input):
    # final relu2
    return flow.math.relu(input,name="finalrelu2")

def finalconv3(input):
    # reshape to (1,1,254,254)
    return _conv2d_layer(
        "finalconv3", input, filters=1, kernel_size=2, strides=1, padding=[0,0,1,1],
    )

def LinkNet34(images,trainable=False,batch_size=1):
    # channel of different filters in different layers
    filters = [64,128,256,512]
    # convolution layer1
    firstconv = _conv2d_layer(name="firstconv", input=images, filters=64, kernel_size=7, strides=2,trainable=trainable,use_bias=False)
    # batch normalisation1
    firstbn = flow.layers.batch_normalization(firstconv, axis=1, momentum=0.1, epsilon=0.00001,name="firstbn")
    # relu
    firstrelu = flow.math.relu(firstbn,name="firstrelu")
    # max pooling layer
    firstmaxpool = flow.nn.max_pool2d(
        firstrelu, ksize=3, strides=2, padding="SAME", data_format="NCHW", name="firstmaxpool",
    )
    # encoder1
    e1 = layer1(firstmaxpool)
    # encoder2
    e2 = layer2(e1)
    # encoder3
    e3 = layer3(e2)
    # encoder4
    e4 = layer4(e3)
    # decoder4
    stage_name = "decoder4-"
    d4 = DecoderBlockLinkNet(e4,filters[3], filters[2],stage_name,d_output_shape=(batch_size,128,16,16)) + e3
    # decoder3
    stage_name = "decoder3-"
    d3 = DecoderBlockLinkNet(d4,filters[2], filters[1],stage_name,d_output_shape=(batch_size,64,32,32)) + e2
    # decoder2
    stage_name = "decoder2-"
    d2 = DecoderBlockLinkNet(d3,filters[1], filters[0],stage_name,d_output_shape=(batch_size,32,64,64)) + e1
    # decoder1
    stage_name = "decoder1-"
    d1 = DecoderBlockLinkNet(d2,filters[0], filters[0],stage_name,d_output_shape=(batch_size,16,128,128))
    # final convolution layer 1
    stage_name = "finaldeconv1"
    f1 = finaldeconv1(input=d1,stage_name=stage_name,output_shape=(batch_size,32,257,257))

    f2 = finalrelu1(f1)
    # final convolution layer 2
    f3 = finalconv2(f2)

    f4 = finalrelu2(f3)

    f5 = finalconv3(f4)

    x_out = f5

    return x_out

def resize_image(img, origin_h, origin_w, image_height, image_width):
    # output width of image
    w = image_width
    # output height of image
    h = image_height

    resized=np.zeros((3, image_height, image_width), dtype=np.float32)
    part=np.zeros((3, origin_h, image_width), dtype = np.float32)
    # scale of width
    w_scale = (float)(origin_w - 1) / (w - 1)
    # scale of height
    h_scale = (float)(origin_h - 1) / (h - 1)

    for c in range(w):
        if c == w-1 or origin_w == 1:
            val = img[:, :, origin_w-1]
        else:
            sx = c * w_scale
            ix = int(sx)
            dx = sx - ix
            val = (1 - dx) * img[:, :, ix] + dx * img[:, :, ix+1]
        part[:, :, c] = val
    for r in range(h):
        sy = r * h_scale
        iy = int(sy)
        dy = sy - iy
        val = (1-dy)*part[:, iy, :]
        resized[:, r, :] = val
        if r==h-1 or origin_h==1:
            continue
        # resized image
        resized[:, r, :] = resized[:, r, :] + dy * part[:, iy+1, :]
    return resized

def batch_image_preprocess(img_path, img_height, img_width):
    result_list = []
    base = np.ones([256,256])
    norm_mean = [base * 0.485, base * 0.456, base * 0.406]  # imagenet mean
    norm_std = [0.229, 0.224, 0.225]  # imagenet std
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = img.transpose(2, 0, 1).astype(np.float32)  # hwc->chw
    img = img / 255  # /255  # to tensor
    img[[0, 1, 2], :, :] = img[[2, 1, 0], :, :]  # bgr2rgb

    w = img_width
    h = img_height
    origin_h = img.shape[1] # orignal height
    origin_w = img.shape[2] # orignal width

    resize_img = resize_image(img, origin_h, origin_w, h, w) # resize img
    # normalize

    resize_img[0] = (resize_img[0] - norm_mean[0]) / norm_std[0]
    resize_img[1] = (resize_img[1] - norm_mean[1]) / norm_std[1]
    resize_img[2] = (resize_img[2] - norm_mean[2]) / norm_std[2]
    result_list.append(resize_img) # image list

    results = np.asarray(result_list).astype(np.float32)

    return results

@flow.global_function(flow.function_config())
def faceseg_job(image=flow.FixedTensorDef((1,3,256,256), dtype=flow.float)):

    feature = LinkNet34(image,trainable=False,batch_size=1) # use linknet34 model to segment face
    return feature

def faceSeg(img_path,model_para_path):

    # input image preprocess
    query_images = batch_image_preprocess(img_path,256,256)

    feature = faceseg_job(query_images).get()
    return feature
