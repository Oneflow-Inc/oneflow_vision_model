import oneflow as flow
import numpy as np
import oneflow.typing as tp
from typing import Tuple
import oneflow.nn as nn
import math
import time
import datetime


class KnowledgeConsistentAttention(object):
    def __init__(self, patch_size=3, propagate_size=3, stride=1,ti=""):
        super(KnowledgeConsistentAttention, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None
        self.att_scores_prev = None
        self.masks_prev = None
        self.time =ti
        # self.ratio = nn.Parameter(torch.ones(1))

    def buildnet(self, foreground, masks):
        #time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')
        bz, nc, w, h = foreground.shape
        if masks.shape[3] != foreground.shape[3]:
            # masks = F.interpolate(masks, foreground.size()[2:])
            if foreground.shape[3] >= masks.shape[3]:
                beishu = (foreground.shape[3]) / (masks.shape[3])
                masks = flow.layers.upsample_2d(masks, size=(beishu, beishu))
            else:
                kk = masks.shape[3] - foreground.shape[3]
                strid = int((masks.shape[3] - 4) / (masks.shape[3] - kk - 1))
                initializer = flow.truncated_normal(0.1)
                masks = flow.layers.conv2d(inputs=masks, filters=masks.shape[1], kernel_size=4,
                                           kernel_initializer=initializer, name="Conv2d" + self.time, padding='VALID',
                                           strides=strid)
        # background = foreground.clone()
        background = foreground
        background = background
        conv_kernels_all = flow.reshape(x=background, shape=(bz, nc, w * h, 1, 1))
        # conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3, 4)
        conv_kernels_all = flow.transpose(a=conv_kernels_all, perm=[0, 2, 1, 3, 4])
        output_tensor = []
        att_score = []
        for i in range(bz):
            time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')
            # feature_map = foreground[i:i+1]
            feature_map = flow.slice(x=foreground, begin=[i, None, None, None], size=[1, None, None, None])
            # conv_kernels = conv_kernels_all[i] + 0.0000001
            conv_kernels = flow.slice(x=conv_kernels_all, begin=[i, None, None, None, None],
                                      size=[1, None, None, None, None])
            conv_kernels = flow.squeeze(input=conv_kernels, axis=[0])
            constant11 = flow.constant_like(like=conv_kernels, value=0.0000001, dtype=flow.float32)
            conv_kernels = flow.math.add(x=conv_kernels, y=constant11)
            # norm_factor = torch.sum(conv_kernels**2, [1, 2, 3], keepdim = True)**0.5
            constant1 = flow.constant_like(like=conv_kernels, value=2, dtype=flow.float32)
            conv_22 = flow.math.pow(x=conv_kernels, y=constant1)
            norm_factor = flow.math.reduce_sum(conv_22, (1, 2, 3), keepdims=True)
            constant = flow.constant_like(like=norm_factor, value=0.5, dtype=flow.float32)
            norm_factor = flow.math.pow(x=norm_factor, y=constant)
            conv_kernels = conv_kernels / norm_factor

            # conv_result = F.conv2d(feature_map, conv_kernels, padding = self.patch_size//2)
            conv_result = flow.nn.conv2d(feature_map, conv_kernels,
                                         padding=[0, 0, self.patch_size // 2, self.patch_size // 2], strides=1,name="nn"+self.time+str(i))
            if self.propagate_size != 1:
                if self.prop_kernels is None:
                    constant_blob = flow.constant(value=1.5, shape=(
                    conv_result.shape[1], 1, self.propagate_size, self.propagate_size), dtype=flow.float)
                    self.prop_kernels = flow.ones_like(like=constant_blob)
                    # self.prop_kernels = torch.ones([conv_result.size(1), 1, self.propagate_size, self.propagate_size])
                    # self.prop_kernels.requires_grad = False
                    # self.prop_kernels = self.prop_kernels.cuda()
                # conv_result = F.avg_pool2d(conv_result, 3, 1, padding = 1)*9
                conv_result = flow.nn.avg_pool2d(conv_result, ksize=3, strides=1, padding=[0, 0, 1, 1]) * 9  #
            attention_scores = flow.nn.softmax(conv_result, axis=1)
            if self.att_scores_prev is not None:
                self.ratio = flow.get_variable(shape=(1), name="ratio", initializer=init, trainable=True)
                # attention_scores = (self.att_scores_prev[i:i+1]*self.masks_prev[i:i+1] + attention_scores * (torch.abs(self.ratio)+1e-7))/(self.masks_prev[i:i+1]+(torch.abs(self.ratio)+1e-7))
                attention_scores = (self.att_scores_prev[i:i + 1] * self.masks_prev[i:i + 1] + attention_scores * (
                        flow.math.abs(self.ratio) + 1e-7)) / (
                                               self.masks_prev[i:i + 1] + (flow.math.abs(self.ratio) + 1e-7))

            att_score.append(attention_scores)
            # feature_map = F.conv_transpose2d(attention_scores, conv_kernels, stride = 1, padding = self.patch_size//2)
            # feature_map = F.conv_transpose2d(attention_scores, conv_kernels, stride=1, padding=self.patch_size // 2)

            weight = conv_kernels

            feature_map = flow.nn.torch_conv2d_transpose(value=attention_scores, filter=weight, padding_needed=(
            self.patch_size // 2 * 2, self.patch_size // 2 * 2), strides=1,
                                                         name="deconv" + self.time +str(i), output_padding=(0, 0))
            final_output = feature_map
            output_tensor.append(final_output)
        self.att_scores_prev = flow.concat(att_score, axis=0)
        self.att_scores_prev = flow.reshape(x=self.att_scores_prev, shape=[bz, h * w, h, w])
        # self.masks_prev = masks.view(bz, 1, h, w)
        self.masks_prev = flow.reshape(x=masks, shape=[bz, 1, h, w])
        # return torch.cat(output_tensor, dim = 0)
        return flow.concat(output_tensor, axis=0)


class AttentionModule(object):

    def __init__(self, inchannel, patch_size_list=[1], propagate_size_list=[3], stride_list=[1],tim=""):
        assert isinstance(patch_size_list,
                          list), "patch_size should be a list containing scales, or you should use Contextual Attention to initialize your module"
        assert len(patch_size_list) == len(propagate_size_list) and len(propagate_size_list) == len(
            stride_list), "the input_lists should have same lengths"
        super(AttentionModule, self).__init__()

        self.inchannel = inchannel
        self.patch_size_list = patch_size_list
        self.propagate_size_list = propagate_size_list
        self.stride_list = stride_list
        self.attt = KnowledgeConsistentAttention(self.patch_size_list[0], self.propagate_size_list[0],
                                                 self.stride_list[0], ti=tim)

        self.num_of_modules = len(patch_size_list)
        self.time =tim
    def buildnet(self, foreground, mask):
        #time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')
        # self.attt = KnowledgeConsistentAttention(self.patch_size_list[0], self.propagate_size_list[0],self.stride_list[0])

        outputs = self.attt.buildnet(foreground, mask)
        outputs = flow.concat(inputs=[outputs, foreground], axis=1)
        # self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)
        # outputs = self.combiner(outputs)
        initializer = flow.truncated_normal(0.1)
        outputs = flow.layers.conv2d(inputs=outputs, filters=self.inchannel, kernel_size=1, name="conv11" + self.time,
                                     kernel_initializer=initializer)
        return outputs

# @flow.global_function()
# def deconv2d_Job(x: tp.Numpy.Placeholder((2, 512, 6, 6)),y: tp.Numpy.Placeholder((2, 1, 6, 6))
# ) -> tp.Numpy:
# deconv = AttentionModule(512)
# output = deconv.buildnet(x,y)
# return output
# x = np.random.randn(2, 512, 6, 6).astype(np.float32)
# y = np.random.randn(2, 1, 6, 6).astype(np.float32)
# out = deconv2d_Job(x,y)
# print(out.shape)

# input =torch.randn(1, 512, 32, 32)
# filter=torch.randn(1, 1, 128, 128)
# deconv =AttentionModule(512)
# output =deconv(input,filter)
# print(output.size())
