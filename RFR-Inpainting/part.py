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
                 padding=0,dilation=1,multi_channel = False, bias = False, *args, **kwargs):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.multi_channel = multi_channel
        self.bias = bias


        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)



        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def buildnet(self, input, mask=None):
        time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')

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


            #第一次卷及操作
            self.update_mask = flow.nn.conv2d(mask, filters=self.weight_maskUpdater, strides=self.stride, padding=[0,0,self.padding,self.padding], groups=1)
            return self.update_mask
            self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                                 self.weight_maskUpdater.shape[3]
            self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
            # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
            self.update_mask = flow.clamp(values=self.update_mask,min_value= 0, max_value =1)
            self.mask_ratio = flow.matmul(self.mask_ratio, self.update_mask)










        #第二次卷积操作
        #raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask is not None else input)
        if mask is not None:

            weight_shape = (self.kernel_size, self.kernel_size)
            init = flow.kaiming_initializer(shape=weight_shape, negative_slope=0, mode='fan_in')
            raw_out = flow.layers.conv2d(flow.matmul(input, mask), kernel_initializer=init, filters=self.out_channels,
                                         kernel_size=self.kernel_size, strides=self.stride,
                                         padding=[0, 0, self.padding, self.padding], name="raw_outconv1" + str(time))
        else:
            weight_shape = (self.kernel_size, self.kernel_size)
            init = flow.kaiming_initializer(shape=weight_shape, negative_slope=0, mode='fan_in')
            raw_out = flow.layers.conv2d(input, filters=self.out_channels, kernel_initializer=init,
                                         kernel_size=self.kernel_size, strides=self.stride,
                                         padding=[0, 0, self.padding, self.padding], name="raw_outconv2" + str(time))

        print(raw_out.numpy())

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
            output = flow.matmul(flow.math.subtract(raw_out, bias_view), self.mask_ratio)
            output = flow.math.add(output, bias_view)
            # output = torch.mul(output, self.update_mask)
            output = flow.matmul(output, self.update_mask)
        else:
            output = flow.matmul(raw_out, self.mask_ratio)
        if self.return_mask:
            return output, self.update_mask
        else:
            return output

# in_image = torch.ones((6, 3, 6, 6))
# mask = torch.ones((6, 3, 6, 6))
# Pconv1 = PartialConv2d(3, 64, 7, 2, 3, multi_channel=True, bias=False)
# x1, m1 = Pconv1(in_image, mask)
# print("image :", x1.shape)
# print("mask :", m1.shape)
# x2 = x1.detach().numpy()
# m2 = m1.detach().numpy()
# print(x2)
# print(m2)



# @flow.global_function()
# def conv2d_Job(
#         x: tp.Numpy.Placeholder((6,3,6,6)),
#         y: tp.Numpy.Placeholder((6,3,6,6))
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
# images = np.ones((6,3,6,6)).astype(np.float32)
# masks = np.ones((6,3,6,6)).astype(np.float32)
#
# out1,out2= conv2d_Job(images,masks)
# print("image :",out1.shape)
# print(out1)
#
# print("mask :",out2.shape)
# print(out2)

@flow.global_function()
def conv2d_Job(
        x: tp.Numpy.Placeholder((6,3,6,6)),
        y: tp.Numpy.Placeholder((6,3,6,6))



) ->tp.Numpy:
    initializer = flow.truncated_normal(0.1)
    conv = PartialConv2d(3, 64, 7, 2, 3, multi_channel = True, bias = False)
    conv1=conv.buildnet(x,y)
    return conv1


images = np.ones((6,3,6,6)).astype(np.float32)
masks = np.ones((6,3,6,6)).astype(np.float32)

out1= conv2d_Job(images,masks)
print("image :",out1.shape)
print(out1)

