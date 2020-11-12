#-*- coding:utf-8 -*-
""" 
 @author: scorpio.lu
 @datetime:2020-06-11 15:24
 @software: PyCharm
 @contact: luyi@zhejianglab.com

            ----------
             路有敬亭山
            ----------
 
"""
import oneflow as flow
import numpy as np
BLOCK_COUNTS = [3, 4, 6, 3]
BLOCK_FILTERS = [256, 512, 1024, 2048]
BLOCK_FILTERS_INNER = [64, 128, 256, 512]

def _conv2d(
    name,
    input,
    filters,
    kernel_size,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilations=1,
    trainable=True,
    weight_initializer=flow.variance_scaling_initializer(data_format="NCHW"),
):
    weight = flow.get_variable(
        name + "-weight",
        shape=(filters, input.shape[1], kernel_size, kernel_size),
        dtype=input.dtype,
        initializer=weight_initializer,
        trainable=trainable,
    )
    return flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilations, name=name
    )

def conv2d_affine(input, name, filters, kernel_size, strides, activation=None, trainable=True):
    # input data_format must be NCHW, cannot check now
    padding = "SAME" if strides > 1 or kernel_size > 1 else "VALID"
    output = _conv2d(name, input, filters, kernel_size, strides, padding, trainable=trainable)
    output = _batch_norm(output, name + "_bn", trainable)
    if activation == "Relu":
        output = flow.math.relu(output)

    return output


def bottleneck_transformation(input, block_name, filters, filters_inner, strides, trainable=True):
    a = conv2d_affine(
        input, block_name + "-branch2a", filters_inner, 1, 1, activation="Relu", trainable=trainable
    )

    b = conv2d_affine(
        a, block_name + "-branch2b", filters_inner, 3, strides, activation="Relu", trainable=trainable
    )

    c = conv2d_affine(b, block_name + "-branch2c", filters, 1, 1, trainable=trainable)

    return c



def _batch_norm(inputs, name=None, trainable=True):
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=1,
        momentum=0.997,
        epsilon=1.001e-5,
        center=True,
        scale=True,
        trainable=trainable,
        name=name,
    )

def layer0(input, trainable):
    conv1 = _conv2d("conv1", input, 64, 7, 2, trainable=trainable)
    conv1_bn = flow.math.relu(_batch_norm(conv1, "bn1", trainable))
    pool1 = flow.nn.max_pool2d(
        conv1_bn, ksize=3, strides=2, padding="VALID", data_format="NCHW", name="pool1",
    )
    return pool1

def resnet_conv_x_body(input, on_stage_end=lambda x: x, trainable=True):
    output = input
    for i, (counts, filters, filters_inner) in enumerate(
        zip(BLOCK_COUNTS, BLOCK_FILTERS, BLOCK_FILTERS_INNER)
    ):
        stage_name = "layer%d" % (i + 1)
        output = residual_stage(
            output, stage_name, counts, filters, filters_inner, 1 if i == 0 or i == 3 else 2, trainable=trainable
        )
        on_stage_end(output)

    return output

def residual_stage(input, stage_name, counts, filters, filters_inner, stride_init=2, trainable=True):
    output = input
    for i in range(counts):
        block_name = "%s-%d" % (stage_name, i)
        output = residual_block(
            output, block_name, filters, filters_inner, stride_init if i == 0 else 1, trainable=trainable
        )

    return output

def residual_block(input, block_name, filters, filters_inner, strides_init, trainable):
    if strides_init != 1 or block_name == "layer1-0" or block_name == "layer4-0":
        shortcut = conv2d_affine(
            input, block_name+'-downsample', filters, 1, strides_init, trainable=trainable
        )
    else:
        shortcut = input

    bottleneck = bottleneck_transformation(
        input, block_name, filters, filters_inner, strides_init, trainable=trainable
    )

    return flow.math.relu(bottleneck + shortcut)


'''
use resnet50 as backbone, modify the stride of last layer to be 1 for rich person features 
'''
def resreid(images, trainable=False):
    with flow.deprecated.variable_scope("base"):
        stem = layer0(images, trainable=trainable)
        body = resnet_conv_x_body(stem, lambda x: x, trainable=trainable)

    with flow.deprecated.variable_scope("gap"):
        pool5 = flow.nn.avg_pool2d(body, ksize=[16, 8], strides=1, padding="VALID", data_format="NCHW", name="pool5")
        feature = flow.reshape(pool5, [-1, int(np.prod(pool5.shape[1:])) ])
    return feature


'''
Slice feature map into two parts horizontally by GAP in order to mining discriminative features
'''

def HS_reid(images, trainable=False):
    with flow.deprecated.variable_scope("base"):
        stem = layer0(images, trainable=trainable)
        body = resnet_conv_x_body(stem, lambda x: x, trainable=trainable)

    with flow.deprecated.variable_scope("gap"):
        pool5 = flow.nn.avg_pool2d(body, ksize=[8, 8], strides=4, padding="VALID", data_format="NCHW", name="pool5")
        #norm = flow.math.pow(pool5, 2)
        #pool5 = flow.math.div(pool5, flow.math.sqrt(flow.math.reduce_sum(flow.math.pow(pool5, 2), axis=1)))
        feature = flow.reshape(pool5, [pool5.shape[0], -1])
    return feature
