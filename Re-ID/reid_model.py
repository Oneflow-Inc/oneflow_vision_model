"""
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# -*- coding:utf-8 -*-
"""
Person Re-Identification models
"""
# Version: 0.0.1
# Author: scorpio.lu(luyi@zhejianglab.com)
# Data: 06/28/2020

import oneflow as flow

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
    """conv2d + batch norm + relu unit"""
    # input data_format must be NCHW
    padding = "SAME" if strides > 1 or kernel_size > 1 else "VALID"
    output = _conv2d(name, input, filters, kernel_size, strides, padding, trainable=trainable)
    output = _batch_norm(output, name + "_bn", trainable)
    if activation == "Relu":
        output = flow.nn.relu(output)

    return output


def bottleneck_transformation(input, block_name, filters, filters_inner, strides, trainable=True):
    """1*1 conv2d_affine + 3*3 conv2d_affine + 1*1 conv2d_affine"""
    a = conv2d_affine(
        input, block_name + "-branch2a", filters_inner, 1, 1, activation="Relu", trainable=trainable
    )

    b = conv2d_affine(
        a, block_name + "-branch2b", filters_inner, 3, strides, activation="Relu", trainable=trainable
    )

    c = conv2d_affine(b, block_name + "-branch2c", filters, 1, 1, trainable=trainable)

    return c


def _batch_norm(inputs, name=None, trainable=True):
    """batch normalization"""
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
    """conv2d + relu + max pooling"""
    conv1 = _conv2d("conv1", input, 64, 7, 2, trainable=trainable)
    conv1_bn = flow.nn.relu(_batch_norm(conv1, "bn1", trainable))
    pool1 = flow.nn.max_pool2d(
        conv1_bn, ksize=3, strides=2, padding="VALID", data_format="NCHW", name="pool1",
    )
    return pool1


def resnet_conv_x_body(input, on_stage_end=lambda x: x, trainable=True):
    """residual blocks of layers"""
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
    """4 layers"""
    output = input
    for i in range(counts):
        block_name = "%s-%d" % (stage_name, i)
        output = residual_block(
            output, block_name, filters, filters_inner, stride_init if i == 0 else 1, trainable=trainable
        )

    return output


def residual_block(input, block_name, filters, filters_inner, strides_init, trainable):
    """a residual block"""
    if strides_init != 1 or block_name == "layer1-0" or block_name == "layer4-0":
        shortcut = conv2d_affine(
            input, block_name + '-downsample', filters, 1, strides_init, trainable=trainable
        )
    else:
        shortcut = input

    bottleneck = bottleneck_transformation(
        input, block_name, filters, filters_inner, strides_init, trainable=trainable
    )

    return flow.nn.relu(bottleneck + shortcut)


def resreid_train(images, num_class=751, trainable=True):
    """use resnet50 as backbone, modify the stride of last layer to be 1 for rich person features """
    with flow.scope.namespace("base"):
        stem = layer0(images, trainable=trainable)
        body = resnet_conv_x_body(stem, lambda x: x, trainable=trainable)
    with flow.scope.namespace("gap"):
        pool5 = flow.nn.avg_pool2d(body, ksize=[16, 8], strides=1, padding="VALID", data_format="NCHW", name="pool5")
        feature = flow.reshape(pool5, [pool5.shape[0], -1])
        if not trainable:
            return feature
        bn1 = flow.layers.batch_normalization(
            feature,
            axis=1,
            center=False,
            beta_initializer=flow.constant_initializer(0),
            gamma_initializer=flow.random_normal_initializer(mean=1, stddev=0.02),
            trainable=trainable,
            name='bnout'
        )
        fc6 = flow.layers.dense(
            inputs=bn1,
            units=num_class,
            activation=None,
            use_bias=False,
            kernel_initializer=flow.random_normal_initializer(mean=0, stddev=0.01),
            trainable=trainable,
            name="fc6",
        )
    return feature, fc6


def HS_reid_train(images, num_class=751, trainable=False):
    """Slice feature map into two parts horizontally by GAP in order to mining discriminative features"""
    with flow.scope.namespace("base"):
        stem = layer0(images, trainable=trainable)
        body = resnet_conv_x_body(stem, lambda x: x, trainable=trainable)

    with flow.scope.namespace("gap"):
        pool5 = flow.nn.avg_pool2d(body, ksize=[4, 8], strides=4, padding="VALID", data_format="NCHW", name="pool5")
        feature = flow.reshape(pool5, [pool5.shape[0], -1])
        if not trainable:
            return feature
        bn1 = flow.layers.batch_normalization(
            feature,
            axis=1,
            center=False,
            beta_initializer=flow.constant_initializer(0),
            gamma_initializer=flow.random_normal_initializer(mean=1, stddev=0.02),
            trainable=trainable,
            name='bnout'
        )
        fc6 = flow.layers.dense(
            inputs=bn1,
            units=num_class,
            activation=None,
            use_bias=False,
            kernel_initializer=flow.random_normal_initializer(mean=0, stddev=0.01),
            trainable=trainable,
            name="fc6",
        )
    return feature, fc6
