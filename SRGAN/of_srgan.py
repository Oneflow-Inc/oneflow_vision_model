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

import os, cv2
import time
import oneflow as flow
from typing import Tuple
import oneflow.typing as tp
import math
import numpy as np
# import pytorch_ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import of_layers as layers
from of_data_utils import load_image, is_image_file
import skimage.metrics
from datetime import datetime
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class SRGAN:
    def __init__(self, args):
        self.lr = args.lr
        self.path = args.path
        self.hr_size = args.hr_size
        self.scale_factor = args.scale_factor
        self.residual_num = args.residual_num
        self.data_dir = args.data_dir
        self.gpu_num_per_node = args.gpu_num_per_node
        self.batch_size = args.batch_size * self.gpu_num_per_node

        self.print_interval = 50
        self.vgg_path = args.vgg_path
        self.lr_size = self.hr_size // self.scale_factor

        if not os.path.exists(self.path):
            os.mkdir(self.path)
            print("Make new dir '{}' done.".format(self.path))
        self.checkpoint_path = os.path.join(self.path, "checkpoint")
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.loss_path = os.path.join(self.path, "loss")
        if not os.path.exists(self.loss_path):
            os.mkdir(self.loss_path)
        # self.train_images_path = os.path.join(self.path, "train_images")
        # if not os.path.exists(self.train_images_path):
        #     os.mkdir(self.train_images_path)
        self.val_images_path = os.path.join(self.path, "val_images")
        if not os.path.exists(self.val_images_path):
            os.mkdir(self.val_images_path)

    def Generator(self, inputs, trainable=True):
        upsample_block_num = int(math.log(self.scale_factor, 2))
        with flow.scope.namespace("g_block1"):
            conv1 = layers.conv2d(inputs,filters=64, size=9, name="conv", trainable=trainable)
            # prelu1 = flow.layers.prelu(conv1, name="prelu")
            relu1 = flow.math.relu(conv1)

        with flow.scope.namespace("g_residual"):
            residual_blocks = layers.residual_blocks(relu1, filters=64, block_num=self.residual_num, trainable=trainable)

        with flow.scope.namespace("g_block7"):
            conv7 = layers.conv2d(residual_blocks, filters=64, size=3, name="conv", trainable=trainable)
            # bn7 = layers.batch_norm(conv7,name="bn", trainable=trainable)
            relu7 = flow.math.relu(conv7)

        with flow.scope.namespace("g_upsample"):
            upsample_blocks = layers.upsample_blocks((relu1+relu7), filters=64, block_num=upsample_block_num, trainable=trainable)
            conv8 = layers.conv2d(upsample_blocks, filters=3, size=9, name="conv", trainable=trainable)
        
        # assert conv8.shape == (self.batch_size, 3, self.hr_size, self.hr_size), "The shape of generated images is {}.".format(conv8.shape)

        return (flow.math.tanh(conv8) + 1)/2

    def Discriminator(self, inputs, trainable=True, reuse=False):
        h = int(np.floor(self.hr_size) // 16)

        conv1 = layers.conv2d(inputs, filters=64, size=3, name="d_conv1", trainable=trainable, reuse=reuse)
        lrelu1 = flow.nn.leaky_relu(conv1, alpha=0.2)
        # print(lrelu1.shape) 

        conv2 = layers.conv2d(lrelu1, filters=64, size=3, name="d_conv2", strides=2, trainable=trainable, reuse=reuse)
        bn2 = layers.batch_norm(conv2, name="d_bn2", trainable=trainable, reuse=reuse)
        lrelu2 = flow.nn.leaky_relu(bn2, alpha=0.2)
        # print(lrelu2.shape) 

        conv3 = layers.conv2d(lrelu2, filters=128, size=3, name="d_conv3", trainable=trainable, reuse=reuse)
        bn3 = layers.batch_norm(conv3, name="d_bn3", trainable=trainable, reuse=reuse)
        lrelu3 = flow.nn.leaky_relu(bn3, alpha=0.2)
        # print(lrelu3.shape) 

        conv4 = layers.conv2d(lrelu3, filters=128, size=3, strides=2, name="d_conv4", trainable=trainable, reuse=reuse)
        bn4 = layers.batch_norm(conv4, name="d_bn4", trainable=trainable, reuse=reuse)
        lrelu4 = flow.nn.leaky_relu(bn4, alpha=0.2)
        # print("lrelu4", lrelu4.shape) 

        conv5 = layers.conv2d(lrelu4, filters=256, size=3, name="d_conv5", trainable=trainable, reuse=reuse)
        bn5 = layers.batch_norm(conv5, name="d_bn5", trainable=trainable, reuse=reuse)
        lrelu5 = flow.nn.leaky_relu(bn5, alpha=0.2)
        # print("lrelu5", lrelu5.shape)

        conv6 = layers.conv2d(lrelu5, filters=256, size=3, strides=2, name="d_conv6", trainable=trainable, reuse=reuse)
        bn6 = layers.batch_norm(conv6, name="d_bn6", trainable=trainable, reuse=reuse)
        lrelu6 = flow.nn.leaky_relu(bn6, alpha=0.2)
        # print("lrelu6", lrelu6.shape)

        conv7 = layers.conv2d(lrelu6, filters=512, size=3, name="d_conv7", trainable=trainable, reuse=reuse)
        bn7 = layers.batch_norm(conv7, name="d_bn7", trainable=trainable, reuse=reuse)
        lrelu7 = flow.nn.leaky_relu(bn7, alpha=0.2)
        # print(lrelu7.shape)

        conv8 = layers.conv2d(lrelu7, filters=512, size=3, strides=2, name="d_conv8", trainable=trainable, reuse=reuse)
        bn8 = layers.batch_norm(conv8, name="d_bn8", trainable=trainable, reuse=reuse)
        lrelu8 = flow.nn.leaky_relu(bn8, alpha=0.2)
        # print(lrelu8.shape)

        pool9 = layers.avg_pool2d(lrelu8, name="d_pool9", size=(h,h), strides=(h,h), padding="VALID", reuse=reuse)
        conv9 = layers.conv2d(pool9, filters=1024, size=1, name="d_conv9", trainable=trainable, reuse=reuse)
        lrelu9 = flow.nn.leaky_relu(conv9, alpha=0.2)
        conv10 = layers.conv2d(lrelu9, filters=1, size=1, name="d_conv10", trainable=trainable, reuse=reuse)
        # print("conv10", conv10.shape)

        return flow.math.sigmoid(flow.reshape(conv10, shape=(self.batch_size, -1)))
    
    def vgg16bn(self, images, trainable=True, need_transpose=False, channel_last=False, training=True, wd=1.0/32768, reuse=False):

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

            weight_regularizer=_get_regularizer(), # weight_decay
            bias_regularizer=_get_regularizer(),
            bn=True,
            reuse=False
        ):  
            name_ = name if reuse == False else name + "_reuse"
            weight_shape = (filters, input.shape[1], kernel_size, kernel_size)

            weight = flow.get_variable(
                name + "_weight",
                shape=weight_shape,
                dtype=input.dtype,
                initializer=weight_initializer,
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

        def _conv_block(in_blob, index, filters, conv_times, reuse=False):
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
                    reuse=reuse
                )

                conv_block.append(conv_i)
                index += 1

            return conv_block

        if need_transpose:
            images = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])
        if channel_last:
            # if channel_last=True, then change mode from 'nchw'to 'nhwc'
            images = flow.transpose(images, name="transpose", perm=[0,2,3,1])
        conv1 = _conv_block(images, 0, 64, 2, reuse=reuse)
        # pool1 = flow.nn.max_pool2d(conv1[-1], 2, 2, "VALID", "NCHW", name="pool1")
        pool1 = layers.max_pool2d(conv1[-1], 2, 2, name="pool1", reuse=reuse)
        
        conv2 = _conv_block(pool1, 2, 128, 2, reuse=reuse)
        # pool2 = flow.nn.max_pool2d(conv2[-1], 2, 2, "VALID", "NCHW", name="pool2")
        pool2 = layers.max_pool2d(conv2[-1], 2, 2,  name="pool2", reuse=reuse)

        conv3 = _conv_block(pool2, 4, 256, 3, reuse=reuse)
        # pool3 = flow.nn.max_pool2d(conv3[-1], 2, 2, "VALID", "NCHW", name="pool3")
        pool3 = layers.max_pool2d(conv3[-1], 2, 2, name="pool3", reuse=reuse)

        conv4 = _conv_block(pool3, 7, 512, 3, reuse=reuse)
        # pool4 = flow.nn.max_pool2d(conv4[-1], 2, 2, 
        # "VALID", "NCHW", name="pool4")
        pool4 = layers.max_pool2d(conv4[-1], 2, 2,  name="pool4", reuse=reuse)

        conv5 = _conv_block(pool4, 10, 512, 3, reuse=reuse)
        # pool5 = flow.nn.max_pool2d(conv5[-1], 2, 2, "VALID", "NCHW", name="pool5")
        pool5 = layers.max_pool2d(conv5[-1], 2, 2,  name="pool5", reuse=reuse)

        return pool5

    def train(self, epochs):
        # download data npy
        train_hr_data_path = os.path.join(self.data_dir, "{}_{}hr_imgs.npy".format("train", self.hr_size))
        train_lr_data_path = os.path.join(self.data_dir, "{}_{}lr_imgs.npy".format("train", self.lr_size))
        val_hr_data_path = os.path.join(self.data_dir, "{}_{}hr_imgs.npy".format("val", self.hr_size))
        val_lr_data_path = os.path.join(self.data_dir, "{}_{}lr_imgs.npy".format("val", self.lr_size))
        
        train_hr_data = np.load(train_hr_data_path)
        train_lr_data = np.load(train_lr_data_path)
        val_hr_data = np.load(val_hr_data_path)
        val_lr_data = np.load(val_lr_data_path)
 
        assert train_hr_data.shape == (16700, 3, self.hr_size, self.hr_size), "The shape of train_hr_data is {}".format(train_hr_data.shape)
        assert val_lr_data.shape == (425, 3, self.lr_size, self.lr_size), "The shape of val_lr_data is {}".format(val_lr_data.shape)

        # save loss
        G_l2_loss = []
        G_gan_loss = []
        G_perceptual_loss = []
        G_tv_loss = []
        G_total_loss = []
        D_total_loss = []
        Val_l2_error = []
        Val_ssim = []
        Val_psnr = []

        # config
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        flow.config.gpu_device_num(self.gpu_num_per_node)
        # train config
        lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [self.lr])

        @flow.global_function(type="predict", function_config=func_config)
        def eval_generator(
            input: tp.Numpy.Placeholder((self.batch_size, 3, self.lr_size, self.lr_size))
            ) -> tp.Numpy:
            g_out = self.Generator(input, trainable=False)
            return g_out

        @flow.global_function(type="train", function_config=func_config)
        def train_generator(
            input: tp.Numpy.Placeholder((self.batch_size, 3, self.lr_size, self.lr_size)),
            target: tp.Numpy.Placeholder((self.batch_size, 3, self.hr_size, self.hr_size))
        ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy, tp.Numpy, tp.Numpy, tp.Numpy]:
            g_out = self.Generator(input, trainable=True)
            g_logits = self.Discriminator(g_out, trainable=False)
            # Adversarial Loss 
            g_gan_loss = 0.001 * flow.math.reduce_mean(1 - g_logits)
            # Image Loss
            g_l2_loss = self.mseloss(g_out, target)
            # TV Loss
            g_tv_loss = self.total_variance_loss(g_out, weight=2e-8)
            # Perceptual loss
            def perceptual_loss(fake, real, weight=1.0):
                fake_feature = self.vgg16bn(fake, trainable=False)
                real_feature = self.vgg16bn(real, trainable=False, reuse=True)

                return self.mseloss(fake_feature, real_feature, weight=weight)
            g_perceptual_loss = perceptual_loss(g_out, target, weight=0.006)

            g_total_loss = g_l2_loss + g_gan_loss + g_perceptual_loss + g_tv_loss
            
            flow.optimizer.Adam(lr_scheduler, beta1=0.5, beta2=0.999).minimize(g_total_loss)

            return g_l2_loss, g_gan_loss, g_perceptual_loss, g_tv_loss, g_total_loss, g_out

        @flow.global_function(type="train", function_config=func_config)
        def train_discriminator(
            input: tp.Numpy.Placeholder((self.batch_size, 3, self.lr_size, self.lr_size)),
            target:tp.Numpy.Placeholder((self.batch_size, 3, self.hr_size, self.hr_size))
        ) -> tp.Numpy:
            g_out = self.Generator(input, trainable=False)
            g_logits = self.Discriminator(g_out, trainable=True)
            d_logits = self.Discriminator(target, trainable=True, reuse=True)

            d_loss = 1 - flow.math.reduce_mean(d_logits - g_logits)

            flow.optimizer.Adam(lr_scheduler, beta1=0.5, beta2=0.999).minimize(d_loss)

            return d_loss

        check_point = flow.train.CheckPoint()
        # load trained weight of vgg16bn and initialize automatically GAN model
        check_point.load(self.vgg_path)

        # trained weights of vgg need to be changed, because vgg is used twice like Discriminator. Please use weights in of_vgg16bn_reuse path to load vgg for perceptual loss.
        # check_point.init()
        # check_point.save("vgg_checkpoint")

        batch_num = len(train_hr_data) // self.batch_size
        pre_best, best_psnr = -1, 0
        print("****************** start training *****************")
        for epoch_idx in range(epochs):
            start = time.time()
            print("****************** train  *****************")
            for batch_idx in range(batch_num):
                inputs = train_lr_data[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ].astype(np.float32, order="C")
                target = train_hr_data[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ].astype(np.float32, order="C")
                d_loss = train_discriminator(inputs, target)
                g_l2_loss, g_gan_loss, g_perceptual_loss, g_tv_loss, g_total_loss, g_out = train_generator(inputs, target)

                d_loss = d_loss.mean()
                g_l2_loss = g_l2_loss.mean()
                g_gan_loss = g_gan_loss.mean()
                g_perceptual_loss = g_perceptual_loss.mean()
                g_tv_loss = g_tv_loss.mean()
                g_total_loss = g_total_loss.mean()

                if (batch_idx + 1) % self.print_interval == 0:
                    print(
                    "{}th epoch, {}th batch, g_l2_loss:{}, g_gan_loss:{}, g_perceptual_loss:{}, g_tv_loss:{}, gloss:{}, dloss:{} ".format(epoch_idx + 1, batch_idx + 1, g_l2_loss, g_gan_loss, g_perceptual_loss, g_tv_loss, g_total_loss, d_loss)
                    )

                    G_l2_loss.append(g_l2_loss)
                    G_gan_loss.append(g_gan_loss)
                    G_perceptual_loss.append(g_perceptual_loss)
                    G_tv_loss.append(g_tv_loss)
                    G_total_loss.append(g_total_loss)
                    D_total_loss.append(d_loss)
            
            print("Time for epoch {} is {} sec.".format(epoch_idx + 1, time.time() - start))

            if (epoch_idx + 1) % 1 == 0:
                # save train images
                # self.save_images(g_out, inputs, target, epoch_idx, name="train")

                # save val images, trainable = False
                # and calculate MSE, SSIMs, SSIM, PSNR
                val_l2_error, val_ssim, val_psnr = 0, 0, 0
                val_batch_num = len(val_hr_data) // self.batch_size
                for val_batch_idx in range(val_batch_num):
                    val_inputs = val_lr_data[val_batch_idx * self.batch_size : (val_batch_idx + 1) * self.batch_size].astype(np.float32, order="C")
                    val_target = val_hr_data[val_batch_idx * self.batch_size : (val_batch_idx + 1) * self.batch_size].astype(np.float32, order="C")
                    val_g_out = eval_generator(val_inputs)

                    val_l2_error += (np.square(val_g_out - val_target).mean())
                    val_ssim += self.ssim(val_target.transpose(0, 2, 3, 1), val_g_out.transpose(0, 2, 3, 1))
                    # val_ssims += (pytorch_ssim.ssim(val_g_out, val_target, oneflow=True).item())
                    val_psnr += self.psnr(val_target.transpose(0, 2, 3, 1), val_g_out.transpose(0, 2, 3, 1))
                    
                # save val images
                self.save_images(val_g_out, val_inputs, val_target, epoch_idx, name="val")

                val_l2_error = val_l2_error / val_batch_num
                val_ssim = val_ssim / val_batch_num
                val_psnr = val_psnr / val_batch_num
                # val_psnr = 10 * np.log10(1 / val_l2_error)
                
                Val_l2_error.append(val_l2_error)
                Val_ssim.append(val_ssim)
                Val_psnr.append(val_psnr)
                print("****************** evalute  *****************")
                print(
                    "{}th epoch, {}th batch, val_l2_error:{}, val_ssim:{}, val_psnr:{}.".format(epoch_idx + 1, batch_idx + 1, val_l2_error, val_ssim, val_psnr)
                    )
                if epoch_idx + 1 > 50 and val_psnr > best_psnr:
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

        # save train loss and val error to plot
        np.save(os.path.join(self.loss_path, 'G_l2_loss_{}.npy'.format(epochs)), G_l2_loss)
        np.save(os.path.join(self.loss_path, 'G_gan_loss_{}.npy'.format(epochs)), G_gan_loss)
        np.save(os.path.join(self.loss_path, 'G_perceptual_loss_{}.npy'.format(epochs)), G_perceptual_loss)
        np.save(os.path.join(self.loss_path, 'G_tv_loss_{}.npy'.format(epochs)), G_tv_loss)
        np.save(os.path.join(self.loss_path, 'G_total_loss_{}.npy'.format(epochs)), G_total_loss)
        np.save(os.path.join(self.loss_path, 'D_total_loss_{}.npy'.format(epochs)), D_total_loss)

        np.save(os.path.join(self.loss_path, 'Val_l2_error_{}.npy'.format(epochs)), Val_l2_error)
        np.save(os.path.join(self.loss_path, 'Val_ssim_{}.npy'.format(epochs)), Val_ssim)
        np.save(os.path.join(self.loss_path, 'Val_psnr_{}.npy'.format(epochs)), Val_psnr)
        print("*************** Train {} done ***************** ".format(self.path))

    def save_images(self, images, real_input, target, epoch_idx, name, path=None):
        plt.figure(figsize=(6, 8))
        display_list = list(zip(real_input, target, images))
        # title = ["Input Image", "Ground Truth", "Predicted Image"]
        idx = 1
        row = 4
        # save 4 images of title
        for i in range(self.batch_size):
            dis = display_list[i]
            for j in range(3):
                plt.subplot(row, 6, idx)
                # plt.title(title[j])
                # getting the pixel values between [0, 1] to plot it.
                plt.imshow(np.array(dis[j]).transpose(1, 2, 0))
                plt.axis("off")
                idx = idx + 1

            if idx > row * 6:
                break
        
        if name == "train":
            images_path = self.train_images_path
        elif name == "val":
            images_path = self.val_images_path
        else:
            images_path = path
        
        save_path = os.path.join(images_path, "{}image_{:02d}.png".format(name, epoch_idx + 1))
        plt.savefig(save_path)
        plt.close()

    def save_image(self, image, save_path):
        plt.imsave(save_path, image.transpose(1, 2, 0))
        print("save image in {}".format(save_path))

    def total_variance_loss(self, images, weight):
        assert images.shape == (self.batch_size, 3, self.hr_size, self.hr_size), "The shape of generated images is {}.".format(images.shape)

        def size_num(inputs):
            return inputs.shape[1] * inputs.shape[2] * inputs.shape[3]

        count_h = size_num(flow.slice(images, [None, 0, 1, 0], [None, 3, self.hr_size, self.hr_size]))
        count_w = size_num(flow.slice(images, [None, 0, 0, 1], [None, 3, self.hr_size, self.hr_size]))

        h_tv = flow.math.reduce_sum(flow.math.squared_difference(
            flow.slice(images, [None, 0, 1, 0], [None, 3, self.hr_size, self.hr_size]), 
            flow.slice(images, [None, 0, 0, 0], [None, 3, self.hr_size-1, self.hr_size]))
        )
            
        w_tv = flow.math.reduce_sum(flow.math.squared_difference(
            flow.slice(images, [None, 0, 0, 1], [None, 3, self.hr_size, self.hr_size]), 
            flow.slice(images, [None, 0, 0, 0], [None, 3, self.hr_size, self.hr_size-1]))
        )

        return weight * 2 * (h_tv / count_h + w_tv / count_w) / images.shape[0]

    def mseloss(self, x, y, weight=1.0, mean=True):
        if mean:
            return weight * flow.math.reduce_mean(flow.math.squared_difference(x, y))
        else:
            return weight * flow.math.reduce_sum(flow.math.squared_difference(x, y))

    def psnr(self, image_true, image_test):
        return skimage.metrics.peak_signal_noise_ratio(image_true, image_test)

    def ssim(self, img1, img2):
        if len(img1.shape) == 4:
            return skimage.metrics.structural_similarity(img1, img2, multichannel=True)
        else:
            return skimage.metrics.structural_similarity(img1, img2)

    def test_image(self, image, save_path, model_path, psnr=False, real_image_path=None):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        flow.config.gpu_device_num(self.gpu_num_per_node)
        @flow.global_function(type="predict", function_config=func_config)
        def eval_generator(
            input: tp.Numpy.Placeholder((1, 3, H, W))
            ) -> tp.Numpy:
            g_out = self.Generator(input, trainable=True)
            return g_out

        check_point = flow.train.CheckPoint()
        check_point.load(model_path)
        if image.shape != (1, 3, H, W):
            image = np.expand_dims(image, axis=0)
        result = eval_generator(image)
        self.save_image(result[0], save_path)
            
        if psnr:
            pytorch_result = load_image(real_image_path)[0].transpose(1, 2, 0)
            pytorch_result = np.expand_dims(pytorch_result, axis=0)
            print(self.psnr(pytorch_result, result.transpose(0, 2, 3, 1)))

        flow.clear_default_session()

if __name__ == "__main__":
    os.environ["ENABLE_USER_OP"] = "True"
    import argparse
    parser = argparse.ArgumentParser(description="flags for training SRGAN")
    # Note gpu_num_per_node * batch_size <= 425
    parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
    parser.add_argument("--epochs", type=int, default=200, required=False)
    parser.add_argument("--path", type=str, default="./of_srgan", required=False)
    parser.add_argument("--data_dir", type=str, default="./data", required=False)
    parser.add_argument("--vgg_path", type=str, default="./models/of_vgg16bn_reuse", required=False)
    parser.add_argument("--lr", type=float, default=0.0002, required=False)
    parser.add_argument("--batch_size", type=int, default=128, required=False)
    parser.add_argument('--hr_size', default=88, type=int, help="the size of high-resolution image")
    parser.add_argument('--scale_factor', default=4, type=int, choices=[2, 4, 8],
    help="super resolution upscale factor")
    parser.add_argument('--residual_num', default=5, type=int,
    help="the number of residual blocks in Generator")
    parser.add_argument("--label_smooth", type=float, default=0, required=False)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--test_image", default="", type=str)
    parser.add_argument("--test_images", default="", type=str)

    args = parser.parse_args()
    print(args)
    srgan = SRGAN(args)

    if not args.test:
        # train
        srgan.train(epochs=args.epochs)
    else:
        # test
        model_path = "./models/srgan"
        if args.test_image:
            # test single image
            image_path = args.test_image
            if is_image_file(image_path):
                image, H, W, save_path = load_image(image_path)
                print("the shape of input test image is {}.".format(image.shape))
                srgan.test_image(image, save_path, model_path)
            else:
                print("Please input an image.")
        else:
            # test images with the same/different shape
            images_dir = args.test_images
            for image_path in os.listdir(images_dir):
                path = os.path.join(images_dir, image_path)
                if is_image_file(path):
                    image, H, W, save_path = load_image(path)
                    print("the shape of input test image is {}.".format(image.shape))
                    srgan.test_image(image, save_path, model_path)
                else:
                    continue

  
