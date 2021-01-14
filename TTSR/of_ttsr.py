import os, cv2
import time
import oneflow as flow
from typing import Tuple
import oneflow.typing as tp
import math
import numpy as np
import of_layers as layers
from of_data_utils import TrainSet, TestSet
from datetime import datetime
import shutil
import oneflow.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TTSR:
    def __init__(self, args):
        self.lr = args.lr
        self.path = args.path
        self.gpu_num_per_node = args.gpu_num_per_node
        self.batch_size = args.batch_size * self.gpu_num_per_node

        self.print_interval = 100
        self.vgg_path = args.vgg_path
        self.val_every = 1

        if not os.path.exists(self.path):
            os.mkdir(self.path)
            print("Make new dir '{}' done.".format(self.path))
        self.checkpoint_path = os.path.join(self.path, "checkpoint")
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.loss_path = os.path.join(self.path, "loss")
        if not os.path.exists(self.loss_path):
            os.mkdir(self.loss_path)

    def vgg16bn(self, images, trainable=True, need_transpose=False, channel_last=False, training=True, wd=1.0 / 32768,
                reuse=False):

        def _get_regularizer():
            return flow.regularizers.l2(0.00005)

        def conv2d_layer(
                name,
                input,
                filters,
                kernel_size=3,
                strides=1,
                padding="SAME",
                data_format="NCHW",
                dilation_rate=1,
                activation="Relu",
                use_bias=True,
                weight_initializer=flow.variance_scaling_initializer(2, 'fan_out', 'random_normal', data_format="NCHW"),
                bias_initializer=flow.zeros_initializer(),

                bn=True,
                reuse=False,
                trainable = True
        ):
            name_ = name if reuse == False else name + "_reuse"
            weight_shape = (filters, input.shape[1], kernel_size, kernel_size)

            weight = flow.get_variable(
                name + "_weight",
                shape=weight_shape,
                dtype=input.dtype,
                initializer=weight_initializer,
                trainable=trainable
            )
            output = flow.nn.conv2d(
                input, weight, strides, padding, data_format, dilation_rate, name=name_
            )
            if use_bias:
                bias = flow.get_variable(
                    name + "_bias",
                    shape=(filters,),
                    dtype=input.dtype,
                    initializer=bias_initializer,
                    trainable=trainable
                )
                output = flow.nn.bias_add(output, bias, data_format)

            if activation is not None:
                if activation == "Relu":
                    if bn:
                        # use of_layers(layers) batch_norm
                        output = layers.batch_norm(output, name + "_bn", reuse=reuse)
                        output = flow.nn.relu(output)
                    else:
                        output = flow.nn.relu(output)
                else:
                    raise NotImplementedError

            return output

        def _conv_block(in_blob, index, filters, conv_times, reuse=False, trainable=True):
            conv_block = []
            conv_block.insert(0, in_blob)
            for i in range(conv_times):
                conv_i = conv2d_layer(
                    name="conv{}".format(index),
                    input=conv_block[i],
                    filters=filters,
                    kernel_size=3,
                    strides=1,
                    bn=True,
                    reuse=reuse,
                    trainable=trainable
                )

                conv_block.append(conv_i)
                index += 1

            return conv_block

        if need_transpose:
            images = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])
        if channel_last:
            # if channel_last=True, then change mode from 'nchw'to 'nhwc'
            images = flow.transpose(images, name="transpose", perm=[0, 2, 3, 1])
        conv1 = _conv_block(images, 0, 64, 2, reuse=reuse, trainable=trainable)
        # pool1 = flow.nn.max_pool2d(conv1[-1], 2, 2, "VALID", "NCHW", name="pool1")
        pool1 = layers.max_pool2d(conv1[-1], 2, 2, name="pool1", reuse=reuse)

        conv2 = _conv_block(pool1, 2, 128, 2, reuse=reuse, trainable=trainable)
        # pool2 = flow.nn.max_pool2d(conv2[-1], 2, 2, "VALID", "NCHW", name="pool2")
        pool2 = layers.max_pool2d(conv2[-1], 2, 2, name="pool2", reuse=reuse)

        conv3 = _conv_block(pool2, 4, 256, 1, reuse=reuse, trainable=trainable)

        return conv1[-2], conv2[-2], conv3[-1]

    def LTE(self, inputs, trainable=True):
        rgb_range = 1
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229, 0.224, 0.225)
        meanshift = layers.meanshift(inputs, rgb_range, vgg_mean, vgg_std)
        x_lv1, x_lv2, x_lv3 = self.vgg16bn(meanshift, trainable=trainable)
        return x_lv1, x_lv2, x_lv3

    def unfold(self, input, kernel_size=3, padding=1, stride=1):
        x1 = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        n, c, h, w = x1.shape
        l1 = math.ceil((h - kernel_size + 1) / stride)
        l2 = math.ceil((w - kernel_size + 1) / stride)
        output = []
        for i in range(0, l1):
            for j in range(0, l2):
                x2 = x1[:, :, i * stride: i * stride + kernel_size, j * stride: j * stride + kernel_size]
                x2 = np.reshape(x2, [n, c * kernel_size * kernel_size, 1])
                output.append(x2)
        output = np.concatenate(output, axis=2)
        return output

    def fold(self, input, output_size, kernel_size, padding, stride):
        n = input.shape[0]
        c = input.shape[1] // kernel_size // kernel_size
        h = output_size[0] + 2 * padding
        w = output_size[1] + 2 * padding
        output = np.zeros((n, c, h, w), dtype=np.float32)
        l1 = (h - kernel_size + 1) // stride
        l2 = (w - kernel_size + 1) // stride
        shape = [n, c, kernel_size, kernel_size]
        for i in range(l1):
            for j in range(l2):
                x = input[:, :, i * l2 + j]
                x = np.reshape(x, shape)
                output[:, :, i * stride: i * stride + kernel_size, j * stride: j * stride + kernel_size] = x
        output = output[:, :, padding: h - padding, padding: w - padding]
        return output

    def bis(self, input, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.shape[0]] + [1 if i != 2 else -1 for i in range(1, len(input.shape))]
        index = flow.reshape(index, shape=views)
        index_list = []
        for i in range(input.shape[1]):
            index_list.append(index)
        index = flow.concat(inputs=index_list, axis=1)
        return flow.dim_gather(input, 2, index)

    def mainnet(self, x, S, T_lv3, T_lv2, T_lv1, trainable=True):
        ### shallow feature extraction
        x = layers.SFE(x, trainable=trainable)

        ### stage11
        x11 = x

        ### soft-attention
        x11_res = x11
        x11_res = flow.concat((x11_res, T_lv3), axis=1)
        x11_res = layers.conv3x3(x11_res, 64, "MainNet_0", trainable=trainable)  # F.relu(self.conv11_head(x11_res))
        x11_res = flow.math.multiply(x11_res, S)
        x11 = x11 + x11_res

        x11_res = x11

        x11_res = layers.residual_blocks(x11_res, 16, "rb11_", trainable=trainable)
        x11_res = layers.conv3x3(x11_res, 64, "MainNet_1", trainable=trainable)
        x11 = x11 + x11_res

        ### stage21, 22
        x21 = x11
        x21_res = x21
        x22 = layers.conv3x3(x11, 256, "MainNet_2", trainable=trainable)
        x22 = nn.relu(layers.PixelShuffle(x22, 2))

        ### soft-attention
        x22_res = x22
        x22_res = flow.concat((x22_res, T_lv2), axis=1)
        x22_res = layers.conv3x3(x22_res, 64, "MainNet_3", trainable=trainable)  # F.relu(self.conv22_head(x22_res))
        x22_res = flow.math.multiply(x22_res, flow.layers.upsample_2d(S, (2, 2), interpolation='bilinear', name='upsanple1'))
        x22 = x22 + x22_res

        x22_res = x22

        x21_res, x22_res = layers.CSFI2(x21_res, x22_res, trainable=trainable)

        x21_res = layers.residual_blocks(x21_res, 8, "rb21_", trainable=trainable)
        x22_res = layers.residual_blocks(x22_res, 8, "rb22_", trainable=trainable)

        x21_res = layers.conv3x3(x21_res, 64, "MainNet_4", trainable=trainable)
        x22_res = layers.conv3x3(x22_res, 64, "MainNet_5", trainable=trainable)
        x21 = x21 + x21_res
        x22 = x22 + x22_res

        ### stage31, 32, 33
        x31 = x21
        x31_res = x31
        x32 = x22
        x32_res = x32
        x33 = layers.conv3x3(x22, 256, "MainNet_6", trainable=trainable)
        x33 = nn.relu(layers.PixelShuffle(x33, 2))

        ### soft-attention
        x33_res = x33
        x33_res = flow.concat((x33_res, T_lv1), axis=1)
        x33_res = layers.conv3x3(x33_res, 64, "MainNet_7", trainable=trainable)  # F.relu(self.conv33_head(x33_res))
        x33_res = flow.math.multiply(x33_res, flow.layers.upsample_2d(S, (4, 4), interpolation='bilinear', name='upsanple3'))
        x33 = x33 + x33_res

        x33_res = x33

        x31_res, x32_res, x33_res = layers.CSFI3(x31_res, x32_res, x33_res, trainable=trainable)

        x31_res = layers.residual_blocks(x31_res, 4, "rb31_", trainable=trainable)
        x32_res = layers.residual_blocks(x32_res, 4, "rb32_", trainable=trainable)
        x33_res = layers.residual_blocks(x33_res, 4, "rb33_", trainable=trainable)

        x31_res = layers.conv3x3(x31_res, 64, "MainNet_8", trainable=trainable)
        x32_res = layers.conv3x3(x32_res, 64, "MainNet_9", trainable=trainable)
        x33_res = layers.conv3x3(x33_res, 64, "MainNet_10", trainable=trainable)
        x31 = x31 + x31_res
        x32 = x32 + x32_res
        x33 = x33 + x33_res

        x = layers.MergeTail(x31, x32, x33, trainable=trainable)

        return x

    def train(self, epochs):
        # download data
        train_data = TrainSet(args)
        val_data = TestSet(args)

        # save loss, psnr, ssim
        Loss = []
        Val_psnr = []
        Val_ssim = []

        # config
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.double)
        flow.config.gpu_device_num(self.gpu_num_per_node)
        flow.config.enable_debug_mode(True)
        # train config
        lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [self.lr])

        @flow.global_function(type="predict", function_config=func_config)
        def train_lte(
            input: tp.Numpy.Placeholder((self.batch_size, 3, 160, 160))
        ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy]:
            x_lv1, x_lv2, x_lv3 = self.LTE(input, trainable=True)
            return x_lv1, x_lv2, x_lv3

        @flow.global_function(type="predict", function_config=func_config)
        def train_searchtransfer(
            lrsr_lv3_unfold: tp.Numpy.Placeholder((self.batch_size, 2304, 1600)),
            refsr_lv3_unfold: tp.Numpy.Placeholder((self.batch_size, 2304, 1600)),
            ref_lv3_unfold: tp.Numpy.Placeholder((self.batch_size, 2304, 1600)),
            ref_lv2_unfold: tp.Numpy.Placeholder((self.batch_size, 4608, 1600)),
            ref_lv1_unfold: tp.Numpy.Placeholder((self.batch_size, 9216, 1600))
        ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy, tp.Numpy]:
            refsr_lv3_unfold = flow.transpose(refsr_lv3_unfold, perm=[0, 2, 1])

            refsr_lv3_unfold = flow.math.l2_normalize(refsr_lv3_unfold, axis=2)  # [N, Hr*Wr, C*k*k]
            lrsr_lv3_unfold = flow.math.l2_normalize(lrsr_lv3_unfold, axis=1)  # [N, C*k*k, H*W]

            R_lv3 = flow.matmul(refsr_lv3_unfold, lrsr_lv3_unfold)  # [N, Hr*Wr, H*W]
            R_lv3_star = flow.math.reduce_max(R_lv3, axis=1)  # [N, H*W]
            R_lv3_star_arg = flow.math.argmax(R_lv3, axis=1)  # [N, H*W]

            T_lv3_unfold = self.bis(ref_lv3_unfold, R_lv3_star_arg)
            T_lv2_unfold = self.bis(ref_lv2_unfold, R_lv3_star_arg)
            T_lv1_unfold = self.bis(ref_lv1_unfold, R_lv3_star_arg)

            return R_lv3_star, T_lv3_unfold, T_lv2_unfold, T_lv1_unfold

        @flow.global_function(type="train", function_config=func_config)
        def train_mainnet(
            lr: tp.Numpy.Placeholder((self.batch_size, 3, 40, 40)),
            S: tp.Numpy.Placeholder((self.batch_size, 1, 40, 40)),
            T_lv3: tp.Numpy.Placeholder((self.batch_size, 256, 40, 40)),
            T_lv2: tp.Numpy.Placeholder((self.batch_size, 128, 80, 80)),
            T_lv1: tp.Numpy.Placeholder((self.batch_size, 64, 160, 160)),
            hr: tp.Numpy.Placeholder((self.batch_size, 3, 160, 160))
        ) -> tp.Numpy:
            sr = self.mainnet(lr, S, T_lv3, T_lv2, T_lv1, trainable=True)
            loss = flow.math.reduce_mean(flow.math.abs(flow.math.subtract(sr, hr)))
            flow.optimizer.Adam(lr_scheduler, 0.9, 0.999).minimize(loss)
            return loss

        @flow.global_function(type="predict", function_config=func_config)
        def eval_lte(
            input: tp.Numpy.Placeholder((1, 3, 160, 160))
        ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy]:
            x_lv1, x_lv2, x_lv3 = self.LTE(input, trainable=False)
            return x_lv1, x_lv2, x_lv3

        @flow.global_function(type="predict", function_config=func_config)
        def eval_searchtransfer(
            lrsr_lv3_unfold: tp.Numpy.Placeholder((1, 2304, 1600)),
            refsr_lv3_unfold: tp.Numpy.Placeholder((1, 2304, 1600)),
            ref_lv3_unfold: tp.Numpy.Placeholder((1, 2304, 1600)),
            ref_lv2_unfold: tp.Numpy.Placeholder((1, 4608, 1600)),
            ref_lv1_unfold: tp.Numpy.Placeholder((1, 9216, 1600))
        ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy, tp.Numpy]:
            refsr_lv3_unfold = flow.transpose(refsr_lv3_unfold, perm=[0, 2, 1])

            refsr_lv3_unfold = flow.math.l2_normalize(refsr_lv3_unfold, axis=2)  # [N, Hr*Wr, C*k*k]
            lrsr_lv3_unfold = flow.math.l2_normalize(lrsr_lv3_unfold, axis=1)  # [N, C*k*k, H*W]

            R_lv3 = flow.matmul(refsr_lv3_unfold, lrsr_lv3_unfold)  # [N, Hr*Wr, H*W]
            R_lv3_star = flow.math.reduce_max(R_lv3, axis=1)  # [N, H*W]
            R_lv3_star_arg = flow.math.argmax(R_lv3, axis=1)  # [N, H*W]

            T_lv3_unfold = self.bis(ref_lv3_unfold, R_lv3_star_arg)
            T_lv2_unfold = self.bis(ref_lv2_unfold, R_lv3_star_arg)
            T_lv1_unfold = self.bis(ref_lv1_unfold, R_lv3_star_arg)

            return R_lv3_star, T_lv3_unfold, T_lv2_unfold, T_lv1_unfold

        @flow.global_function(type="predict", function_config=func_config)
        def eval_mainnet(
            lr: tp.Numpy.Placeholder((1, 3, 40, 40)),
            S: tp.Numpy.Placeholder((1, 1, 40, 40)),
            T_lv3: tp.Numpy.Placeholder((1, 256, 40, 40)),
            T_lv2: tp.Numpy.Placeholder((1, 128, 80, 80)),
            T_lv1: tp.Numpy.Placeholder((1, 64, 160, 160))
        ) -> tp.Numpy:
            sr = self.mainnet(lr, S, T_lv3, T_lv2, T_lv1, trainable=False)
            return sr

        check_point = flow.train.CheckPoint()
        check_point.load(self.vgg_path)

        batch_num = len(train_data) // self.batch_size
        pre_best, best_psnr = -1, 0
        print("****************** start training *****************")
        for epoch_idx in range(epochs):
            start = time.time()
            train_data.shuffle(epoch_idx)
            print("****************** train  *****************")
            for batch_idx in range(batch_num):
                lr, lr_sr, hr, ref, ref_sr = [], [], [], [], []
                for idx in range(self.batch_size):
                    sample = train_data[batch_idx * self.batch_size + idx]
                    lr.append(sample['LR'][np.newaxis, :])
                    lr_sr.append(sample['LR_sr'][np.newaxis, :])
                    hr.append(sample['HR'][np.newaxis, :])
                    ref.append(sample['Ref'][np.newaxis, :])
                    ref_sr.append(sample['Ref_sr'][np.newaxis, :])
                lr = np.ascontiguousarray(np.concatenate(lr, axis=0))
                lr_sr = np.ascontiguousarray(np.concatenate(lr_sr, axis=0))
                hr = np.ascontiguousarray(np.concatenate(hr, axis=0))
                ref = np.ascontiguousarray(np.concatenate(ref, axis=0))
                ref_sr = np.ascontiguousarray(np.concatenate(ref_sr, axis=0))

                _, _, lrsr_lv3 = train_lte((lr_sr + 1.) / 2.)
                _, _, refsr_lv3 = train_lte((ref_sr + 1.) / 2.)
                ref_lv1, ref_lv2, ref_lv3 = train_lte((ref + 1.) / 2.)

                ### search
                lrsr_lv3_unfold = self.unfold(lrsr_lv3)
                refsr_lv3_unfold = self.unfold(refsr_lv3)

                ### transfer
                ref_lv3_unfold = self.unfold(ref_lv3)
                ref_lv2_unfold = self.unfold(ref_lv2, kernel_size=6, padding=2, stride=2)
                ref_lv1_unfold = self.unfold(ref_lv1, kernel_size=12, padding=4, stride=4)

                R_lv3_star, T_lv3_unfold, T_lv2_unfold, T_lv1_unfold = train_searchtransfer(
                    lrsr_lv3_unfold, refsr_lv3_unfold, ref_lv3_unfold, ref_lv2_unfold, ref_lv1_unfold)

                T_lv3 = self.fold(T_lv3_unfold, output_size=lrsr_lv3.shape[-2:],
                                  kernel_size=3, padding=1, stride=1) / (3. * 3.)
                T_lv2 = self.fold(T_lv2_unfold, output_size=(lrsr_lv3.shape[2] * 2, lrsr_lv3.shape[3] * 2),
                                  kernel_size=6, padding=2, stride=2) / (3. * 3.)
                T_lv1 = self.fold(T_lv1_unfold, output_size=(lrsr_lv3.shape[2] * 4, lrsr_lv3.shape[3] * 4),
                                  kernel_size=12, padding=4, stride=4) / (3. * 3.)

                S = np.reshape(R_lv3_star, [R_lv3_star.shape[0], 1, lrsr_lv3.shape[2], lrsr_lv3.shape[3]])

                loss = train_mainnet(lr, S, T_lv3, T_lv2, T_lv1, hr)

                if (batch_idx + 1) % self.print_interval == 0:
                    print("{}th epoch, {}th batch, loss:{} ".format(epoch_idx + 1, batch_idx + 1, loss))

                    Loss.append(loss)

            print("Time for epoch {} is {} sec.".format(epoch_idx + 1, time.time() - start))

            if (epoch_idx + 1) % self.val_every == 0:
                val_psnr, val_ssim = 0., 0.
                val_batch_num = len(val_data)
                for batch_idx in range(val_batch_num):
                    sample = val_data[batch_idx]
                    lr = np.ascontiguousarray(sample['LR'][np.newaxis, :])
                    lr_sr = np.ascontiguousarray(sample['LR_sr'][np.newaxis, :])
                    hr = np.ascontiguousarray(sample['HR'][np.newaxis, :])
                    ref = np.ascontiguousarray(sample['Ref'][np.newaxis, :])
                    ref_sr = np.ascontiguousarray(sample['Ref_sr'][np.newaxis, :])

                    _, _, lrsr_lv3 = eval_lte((lr_sr + 1.) / 2.)
                    _, _, refsr_lv3 = eval_lte((ref_sr + 1.) / 2.)
                    ref_lv1, ref_lv2, ref_lv3 = eval_lte((ref + 1.) / 2.)

                    ### search
                    lrsr_lv3_unfold = self.unfold(lrsr_lv3)
                    refsr_lv3_unfold = self.unfold(refsr_lv3)

                    ### transfer
                    ref_lv3_unfold = self.unfold(ref_lv3)
                    ref_lv2_unfold = self.unfold(ref_lv2, kernel_size=6, padding=2, stride=2)
                    ref_lv1_unfold = self.unfold(ref_lv1, kernel_size=12, padding=4, stride=4)

                    R_lv3_star, T_lv3_unfold, T_lv2_unfold, T_lv1_unfold = eval_searchtransfer(
                        lrsr_lv3_unfold, refsr_lv3_unfold, ref_lv3_unfold, ref_lv2_unfold, ref_lv1_unfold)

                    T_lv3 = self.fold(T_lv3_unfold, output_size=lrsr_lv3.shape[-2:],
                                      kernel_size=3, padding=1, stride=1) / (3. * 3.)
                    T_lv2 = self.fold(T_lv2_unfold, output_size=(lrsr_lv3.shape[2] * 2, lrsr_lv3.shape[3] * 2),
                                      kernel_size=6, padding=2, stride=2) / (3. * 3.)
                    T_lv1 = self.fold(T_lv1_unfold, output_size=(lrsr_lv3.shape[2] * 4, lrsr_lv3.shape[3] * 4),
                                      kernel_size=12, padding=4, stride=4) / (3. * 3.)

                    S = np.reshape(R_lv3_star, [R_lv3_star.shape[0], 1, lrsr_lv3.shape[2], lrsr_lv3.shape[3]])

                    sr = eval_mainnet(lr, S, T_lv3, T_lv2, T_lv1)
                    # sr: range [-1, 1]
                    # hr: range [-1, 1]

                    ### prepare data
                    sr = (sr + 1.) * 127.5
                    hr = (hr + 1.) * 127.5

                    sr = np.transpose(np.round(np.squeeze(sr)), (1, 2, 0))
                    hr = np.transpose(np.round(np.squeeze(hr)), (1, 2, 0))

                    ### calculate psnr and ssim
                    val_psnr += self.calc_psnr(sr, hr)
                    val_ssim += self.calc_ssim(sr, hr)

                val_psnr = val_psnr / val_batch_num
                val_ssim = val_ssim / val_batch_num

                Val_psnr.append(val_psnr)
                Val_ssim.append(val_ssim)
                print("****************** evalute  *****************")
                print("{}th epoch, val_psnr:{}, val_ssim:{}.".format(epoch_idx + 1, val_psnr, val_ssim))
                if epoch_idx + 1 > 10 and val_psnr > best_psnr:
                    best_psnr = val_psnr
                    if pre_best != -1:
                        # delete the previous best checkpoint
                        print("delete the previous best {}th epoch model".format(pre_best))
                        shutil.rmtree(os.path.join(self.checkpoint_path, "{}th_epoch".format(pre_best)))

                    # save parameters
                    check_point.save(
                        os.path.join(self.checkpoint_path, "{}th_epoch".format(epoch_idx + 1))
                    )
                    pre_best = epoch_idx + 1
                    print("save the best {}th epoch model at {}.".format(epoch_idx + 1, str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))

        # save train loss and val
        np.save(os.path.join(self.loss_path, 'loss_{}.npy'.format(epochs)), Loss)

        np.save(os.path.join(self.loss_path, 'Val_psnr_{}.npy'.format(epochs)), Val_psnr)
        np.save(os.path.join(self.loss_path, 'Val_ssim_{}.npy'.format(epochs)), Val_ssim)
        print("*************** Train {} done ***************** ".format(self.path))

    def test(self, model_path):
        # download data
        val_data = TestSet(args)

        # config
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.double)
        flow.config.gpu_device_num(self.gpu_num_per_node)
        flow.config.enable_debug_mode(True)

        @flow.global_function(type="predict", function_config=func_config)
        def eval_lte(
            input: tp.Numpy.Placeholder((1, 3, 160, 160))
        ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy]:
            x_lv1, x_lv2, x_lv3 = self.LTE(input, trainable=False)
            return x_lv1, x_lv2, x_lv3

        @flow.global_function(type="predict", function_config=func_config)
        def eval_searchtransfer(
            lrsr_lv3_unfold: tp.Numpy.Placeholder((1, 2304, 1600)),
            refsr_lv3_unfold: tp.Numpy.Placeholder((1, 2304, 1600)),
            ref_lv3_unfold: tp.Numpy.Placeholder((1, 2304, 1600)),
            ref_lv2_unfold: tp.Numpy.Placeholder((1, 4608, 1600)),
            ref_lv1_unfold: tp.Numpy.Placeholder((1, 9216, 1600))
        ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy, tp.Numpy]:
            refsr_lv3_unfold = flow.transpose(refsr_lv3_unfold, perm=[0, 2, 1])

            refsr_lv3_unfold = flow.math.l2_normalize(refsr_lv3_unfold, axis=2)  # [N, Hr*Wr, C*k*k]
            lrsr_lv3_unfold = flow.math.l2_normalize(lrsr_lv3_unfold, axis=1)  # [N, C*k*k, H*W]

            R_lv3 = flow.matmul(refsr_lv3_unfold, lrsr_lv3_unfold)  # [N, Hr*Wr, H*W]
            R_lv3_star = flow.math.reduce_max(R_lv3, axis=1)  # [N, H*W]
            R_lv3_star_arg = flow.math.argmax(R_lv3, axis=1)  # [N, H*W]

            T_lv3_unfold = self.bis(ref_lv3_unfold, R_lv3_star_arg)
            T_lv2_unfold = self.bis(ref_lv2_unfold, R_lv3_star_arg)
            T_lv1_unfold = self.bis(ref_lv1_unfold, R_lv3_star_arg)

            return R_lv3_star, T_lv3_unfold, T_lv2_unfold, T_lv1_unfold

        @flow.global_function(type="predict", function_config=func_config)
        def eval_mainnet(
            lr: tp.Numpy.Placeholder((1, 3, 40, 40)),
            S: tp.Numpy.Placeholder((1, 1, 40, 40)),
            T_lv3: tp.Numpy.Placeholder((1, 256, 40, 40)),
            T_lv2: tp.Numpy.Placeholder((1, 128, 80, 80)),
            T_lv1: tp.Numpy.Placeholder((1, 64, 160, 160))
        ) -> tp.Numpy:
            sr = self.mainnet(lr, S, T_lv3, T_lv2, T_lv1, trainable=False)
            return sr

        check_point = flow.train.CheckPoint()
        check_point.load(model_path)

        val_psnr, val_ssim = 0., 0.
        val_batch_num = len(val_data)
        for batch_idx in range(val_batch_num):
            sample = val_data[batch_idx]
            lr = np.ascontiguousarray(sample['LR'][np.newaxis, :])
            lr_sr = np.ascontiguousarray(sample['LR_sr'][np.newaxis, :])
            hr = np.ascontiguousarray(sample['HR'][np.newaxis, :])
            ref = np.ascontiguousarray(sample['Ref'][np.newaxis, :])
            ref_sr = np.ascontiguousarray(sample['Ref_sr'][np.newaxis, :])

            _, _, lrsr_lv3 = eval_lte((lr_sr + 1.) / 2.)
            _, _, refsr_lv3 = eval_lte((ref_sr + 1.) / 2.)
            ref_lv1, ref_lv2, ref_lv3 = eval_lte((ref + 1.) / 2.)

            ### search
            lrsr_lv3_unfold = self.unfold(lrsr_lv3)
            refsr_lv3_unfold = self.unfold(refsr_lv3)

            ### transfer
            ref_lv3_unfold = self.unfold(ref_lv3)
            ref_lv2_unfold = self.unfold(ref_lv2, kernel_size=6, padding=2, stride=2)
            ref_lv1_unfold = self.unfold(ref_lv1, kernel_size=12, padding=4, stride=4)

            R_lv3_star, T_lv3_unfold, T_lv2_unfold, T_lv1_unfold = eval_searchtransfer(
                lrsr_lv3_unfold, refsr_lv3_unfold, ref_lv3_unfold, ref_lv2_unfold, ref_lv1_unfold)

            T_lv3 = self.fold(T_lv3_unfold, output_size=lrsr_lv3.shape[-2:],
                              kernel_size=3, padding=1, stride=1) / (3. * 3.)
            T_lv2 = self.fold(T_lv2_unfold, output_size=(lrsr_lv3.shape[2] * 2, lrsr_lv3.shape[3] * 2),
                              kernel_size=6, padding=2, stride=2) / (3. * 3.)
            T_lv1 = self.fold(T_lv1_unfold, output_size=(lrsr_lv3.shape[2] * 4, lrsr_lv3.shape[3] * 4),
                              kernel_size=12, padding=4, stride=4) / (3. * 3.)

            S = np.reshape(R_lv3_star, [R_lv3_star.shape[0], 1, lrsr_lv3.shape[2], lrsr_lv3.shape[3]])

            sr = eval_mainnet(lr, S, T_lv3, T_lv2, T_lv1)
            # sr: range [-1, 1]
            # hr: range [-1, 1]

            ### prepare data
            sr = (sr + 1.) * 127.5
            hr = (hr + 1.) * 127.5

            sr = np.transpose(np.round(np.squeeze(sr)), (1, 2, 0))
            hr = np.transpose(np.round(np.squeeze(hr)), (1, 2, 0))

            ### calculate psnr and ssim
            val_psnr += self.calc_psnr(sr, hr)
            val_ssim += self.calc_ssim(sr, hr)

        val_psnr = val_psnr / val_batch_num
        val_ssim = val_ssim / val_batch_num

        print("****************** evalute  *****************")
        print("val_psnr:{}, val_ssim:{}.".format(val_psnr, val_ssim))

    def calc_psnr(self, img1, img2):
        ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
        diff = (img1 - img2) / 255.0
        diff[:, :, 0] = diff[:, :, 0] * 65.738 / 256.0
        diff[:, :, 1] = diff[:, :, 1] * 129.057 / 256.0
        diff[:, :, 2] = diff[:, :, 2] * 25.064 / 256.0

        diff = np.sum(diff, axis=2)
        mse = np.mean(np.power(diff, 2))
        return -10 * math.log10(mse)

    def calc_ssim(self, img1, img2):
        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
        # the same outputs as MATLAB's
        border = 0
        img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 256.0 + 16.0
        img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 256.0 + 16.0
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        h, w = img1.shape[:2]
        img1_y = img1_y[border:h - border, border:w - border]
        img2_y = img2_y[border:h - border, border:w - border]

        if img1_y.ndim == 2:
            return ssim(img1_y, img2_y)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="flags for training TTSR")
    parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
    parser.add_argument("--epochs", type=int, default=100, required=False)
    parser.add_argument("--path", type=str, default="./models", required=False)
    parser.add_argument("--data_dir", type=str, default="./data", required=False)
    parser.add_argument("--vgg_path", type=str, default="./models/of_vgg16bn_reuse", required=False)
    parser.add_argument("--lr", type=float, default=1e-4, required=False)
    parser.add_argument("--batch_size", type=int, default=9, required=False)
    parser.add_argument("--test", action='store_true', default=False)

    args = parser.parse_args()
    print(args)
    ttsr = TTSR(args)

    if not args.test:
        # train
        ttsr.train(epochs=args.epochs)
    else:
        # test
        model_path = "./models/of_ttsr_best_checkpoints"
        ttsr.test(model_path)