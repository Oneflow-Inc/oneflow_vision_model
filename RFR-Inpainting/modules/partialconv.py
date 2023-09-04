###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

#import torch
#import torch.nn.functional as F
#from torch import nn
import numpy as np
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple
import datetime

###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################


import numpy as np
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple
import datetime

def _get_bias_initializer():
    return flow.zeros_initializer()

class PartialConv2d(object):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1,
                 padding=0,dilation=1,multi_channel = False,bias = False, time="",*args, **kwargs):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.multi_channel = multi_channel
        self.bias = bias
        self.time = time

        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)



        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def buildnet(self, input, mask=None):
        #time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')

        if self.multi_channel:
            constant_blob = flow.constant(value=1.5, shape=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size), dtype=flow.float)
            self.weight_maskUpdater = flow.ones_like(like=constant_blob, dtype=input.dtype)
        else:
            constant_blob = flow.constant(value=1.5, shape=(1, 1, self.kernel_size, self.kernel_size), dtype=flow.float)
            self.weight_maskUpdater = flow.ones_like(like=constant_blob, dtype=input.dtype)

        if mask is not None or self.last_size != (input.shape[2], input.shape[3]):
            self.last_size = (input.shape[2], input.shape[3])





            if mask is None:
                if self.multi_channel:
                    cons_blob = flow.constant(value=1.0, shape=(
                    input.shape[0], input.shape[1], input.shape[2], input.shape[3]), dtype=flow.float)
                    mask = flow.ones_like(like=cons_blob, dtype=input.dtype)

                else:
                    cons_blob = flow.constant(value=1.0,
                                                shape=(1, 1, input.shape[2], input.shape[3]),
                                                dtype=flow.float)
                    mask = flow.ones_like(like=cons_blob, dtype=input.dtype)


            #第一次卷积操作
            self.update_mask = flow.nn.conv2d(mask, filters=self.weight_maskUpdater, strides=self.stride, padding=[0,0,self.padding,self.padding], groups=1,name=self.time+"updatemask")

            self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                                 self.weight_maskUpdater.shape[3]
            self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
            # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
            self.update_mask = flow.clamp(values=self.update_mask,min_value= 0, max_value =1)
            self.mask_ratio = flow.math.multiply(self.mask_ratio, self.update_mask)










        #第二次卷积操作
        #raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask is not None else input)
        if mask is not None:

            weight_shape = (self.kernel_size, self.kernel_size)
            init = flow.kaiming_initializer(shape=weight_shape, negative_slope=5.0, mode='fan_in',nonlinearity="leaky_relu",distribution='random_uniform')
            raw_out = flow.layers.conv2d(flow.math.multiply(input, mask), kernel_initializer=init, filters=self.out_channels,
                                         kernel_size=self.kernel_size, strides=self.stride,
                                         padding=[0, 0, self.padding, self.padding], name="raw_outconv1" + self.time)
        else:
            weight_shape = (self.kernel_size, self.kernel_size)
            init = flow.kaiming_initializer(shape=weight_shape, negative_slope=5.0, mode='fan_in',nonlinearity="leaky_relu",distribution='random_uniform')
            raw_out = flow.layers.conv2d(input, filters=self.out_channels, kernel_initializer=init,
                                         kernel_size=self.kernel_size, strides=self.stride,
                                         padding=[0, 0, self.padding, self.padding], name="raw_outconv2" + self.time)


        if self.bias is True:
            # bias_view = self.bias.view(1, self.out_channels, 1, 1)
            # bias_view = flow.reshape(x=self.bias,shape=(1, self.out_channels, 1, 1))
            bias_view = flow.get_variable(
                "bias_view",
                shape=(1, self.out_channels, 1, 1),
                dtype=input.dtype,
                initializer=_get_bias_initializer(),

            )
            # output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = flow.math.multiply(flow.math.subtract(raw_out, bias_view), self.mask_ratio)
            output = flow.math.add(output, bias_view)
            # output = torch.mul(output, self.update_mask)
            output = flow.math.multiply(output, self.update_mask)
        else:
            output = flow.math.multiply(raw_out, self.mask_ratio)
        if self.return_mask:
            return output, self.update_mask
        else:
            return output

#@flow.global_function()
#def conv2d_Job(
        #x: tp.Numpy.Placeholder((2, 256, 6, 6))



#) ->Tuple[tp.Numpy, tp.Numpy]:
    #initializer = flow.truncated_normal(0.1)
    #conv = PartialConv2d(
        #in_channels=256, out_channels=256, kernel_size=3, stride=1,
        #padding=0, multi_channel=True, bias=False,
    #)
    #conv1,conv2=conv.buildnet(x)
    #return conv1,conv2


#x = np.random.randn(2, 256, 6, 6).astype(np.float32)

#out1,out2= conv2d_Job(x)

#print(out1)
#print(out2)


#input =torch.rand(2,256,6,6)
#conv=PartialConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
        #padding=0, multi_channel=True, bias=False,)
#conv1,conv2=conv(input)
#print(conv1.size())
#print(conv2.size())
# @flow.global_function()
# def conv2d_Job(
#         x: tp.Numpy.Placeholder((6,3,256,256)),
#         y: tp.Numpy.Placeholder((6,3,256,256))
#
#
#
# ) ->Tuple[tp.Numpy, tp.Numpy]:
#     initializer = flow.truncated_normal(0.1)
#     conv = PartialConv2d(3, 64, 7, 2, 3, multi_channel = True, bias = False)
#     conv1,conv2=conv.buildnet(x,y)
#     return conv1,conv2
#
#
# images = np.ones((6,3,256,256)).astype(np.float32)
# masks = np.ones((6,3,256,256)).astype(np.float32)
#
# out1,out2= conv2d_Job(images,masks)
# print("image :",out1.shape)
# print(out1)
#
# print("mask :",out2.shape)
# print(out2)