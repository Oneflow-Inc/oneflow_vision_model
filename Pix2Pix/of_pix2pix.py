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

import oneflow as flow
from typing import Tuple
import oneflow.typing as tp
import numpy as np
import imageio
import os
import pix_layers as layers
import matplotlib.pyplot as plt
import time, shutil
import glob
from datetime import datetime
import logging
from io import TextIOBase
import sys

logger = logging.getLogger(__name__)
log_level_map = {
    'fatal': logging.FATAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

_time_format = '%m/%d/%Y, %I:%M:%S %p'

class _LoggerFileWrapper(TextIOBase):
    def __init__(self, logger_file):
        self.file = logger_file

    def write(self, s):
        if s != '\n':
            cur_time = datetime.now().strftime(_time_format)
            self.file.write('[{}] PRINT '.format(cur_time) + s + '\n')
            self.file.flush()
        return len(s)

def init_logger(logger_file_path, log_level_name='info'):
    """Initialize root logger.
    This will redirect anything from logging.getLogger() as well as stdout to specified file.
    logger_file_path: path of logger file (path-like object).
    """
    
    log_level = log_level_map.get(log_level_name)
    logger_file = open(logger_file_path, 'w')
    fmt = '[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s'
    logging.Formatter.converter = time.localtime
    formatter = logging.Formatter(fmt, _time_format)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(logger_file_path)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)

    # include print function output
    sys.stdout = _LoggerFileWrapper(logger_file)

def mkdirs(*args):
    for path in args:
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            logger.info("Make {} in dir: {}".format(path, dirname))
            os.makedirs(dirname)

class Pix2Pix:
    def __init__(self, args):
        self.lr = args.learning_rate
        self.out_channels = 3
        self.img_size = 256
        self.eval_interval = 10
        self.LAMBDA = args.LAMBDA
        self.label_smooth = args.label_smooth
        self.gpus_per_node = args.gpu_num_per_node
        self.batch_size = args.batch_size * self.gpus_per_node
        self.path = args.train_out
        # self.checkpoint_path = os.path.join(self.path, "checkpoint")
        # if not os.path.exists(self.checkpoint_path):
        #     os.mkdir(self.checkpoint_path)
        # self.test_images_path = os.path.join(self.path, "test_images")
        # if not os.path.exists(self.test_images_path):
        #     os.mkdir(self.test_images_path)

    def _downsample(
        self,
        inputs,
        filters,
        size,
        name,
        reuse=False,
        apply_batchnorm=True,
        trainable=True,
        const_init=True,
    ):
        out = layers.conv2d(
            inputs,
            filters,
            size,
            const_init=const_init,
            reuse=reuse,
            trainable=trainable,
            use_bias=False,
            name=name + "_conv",
        )

        if apply_batchnorm:  #and not const_init:
            out = layers.batchnorm(out, name=name + "_bn", reuse=reuse, trainable=trainable)

        out = flow.nn.leaky_relu(out, alpha=0.3)
        return out

    def _upsample(
        self,
        inputs,
        filters,
        size,
        name,
        apply_dropout=False,
        trainable=True,
        const_init=True,
        reuse=False,
    ):
        out = layers.deconv2d(
            inputs,
            filters,
            size,
            const_init=const_init,
            trainable=trainable,
            use_bias=False,
            name=name + "_deconv",
        )

        # out = layers.batchnorm(out, name=name + "_bn", trainable=trainable)
        out = layers.batchnorm(out, name=name + "_bn", reuse=reuse, trainable=trainable)

        if apply_dropout:
            out = flow.nn.dropout(out, rate=0.5)

        out = flow.nn.relu(out)
        return out

    def generator(self, inputs, trainable=True, const_init=False):
        if const_init:
            apply_dropout = False
        else:
            apply_dropout = True
        # (n, 64, 128, 128)
        d1 = self._downsample(
            inputs,
            64,
            4,
            const_init=const_init,
            trainable=trainable,
            apply_batchnorm=False,
            name="g_d1",
        )
        # (n, 128, 64, 64)
        d2 = self._downsample(
            d1, 128, 4, const_init=const_init, trainable=trainable, name="g_d2"
        )
        # (n, 256, 32, 32)
        d3 = self._downsample(
            d2, 256, 4, const_init=const_init, trainable=trainable, name="g_d3"
        )
        # (n, 512, 16, 16)
        d4 = self._downsample(
            d3, 512, 4, const_init=const_init, trainable=trainable, name="g_d4"
        )
        # (n, 512, 8, 8)
        d5 = self._downsample(
            d4, 512, 4, const_init=const_init, trainable=trainable, name="g_d5"
        )
        # (n, 512, 4, 4)
        d6 = self._downsample(
            d5, 512, 4, const_init=const_init, trainable=trainable, name="g_d6"
        )
        # (n, 512, 2, 2)
        d7 = self._downsample(
            d6, 512, 4, const_init=const_init, trainable=trainable, name="g_d7"
        )
        # (n, 512, 1, 1)
        d8 = self._downsample(
            d7, 512, 4, const_init=const_init, trainable=trainable, name="g_d8"
        )
        # (n, 1024, 2, 2)
        u7 = self._upsample(
            d8,
            512,
            4,
            const_init=const_init,
            trainable=trainable,
            apply_dropout=apply_dropout,
            name="g_u7",
        )
        u7 = flow.concat([u7, d7], axis=1)
        # (n, 1024, 4, 4)
        u6 = self._upsample(
            u7,
            512,
            4,
            const_init=const_init,
            trainable=trainable,
            apply_dropout=apply_dropout,
            name="g_u6",
        )
        u6 = flow.concat([u6, d6], axis=1)
        # (n, 1024, 8, 8)
        u5 = self._upsample(
            u6,
            512,
            4,
            const_init=const_init,
            trainable=trainable,
            apply_dropout=apply_dropout,
            name="g_u5",
        )
        u5 = flow.concat([u5, d5], axis=1)
        # (n, 1024, 16, 16)
        u4 = self._upsample(
            u5, 512, 4, const_init=const_init, trainable=trainable, name="g_u4"
        )
        u4 = flow.concat([u4, d4], axis=1)
        # (n, 512, 32, 32)
        u3 = self._upsample(
            u4, 256, 4, const_init=const_init, trainable=trainable, name="g_u3"
        )
        u3 = flow.concat([u3, d3], axis=1)
        # (n, 256, 64, 64)
        u2 = self._upsample(
            u3, 128, 4, const_init=const_init, trainable=trainable, name="g_u2"
        )
        u2 = flow.concat([u2, d2], axis=1)
        # (n, 128, 128, 128)
        u1 = self._upsample(
            u2, 64, 4, const_init=const_init, trainable=trainable, name="g_u1"
        )
        u1 = flow.concat([u1, d1], axis=1)
        # (n, 3, 256, 256)
        u0 = layers.deconv2d(
            u1,
            self.out_channels,
            4,
            name="g_u0_deconv",
            const_init=const_init,
            trainable=trainable,
        )
        u0 = flow.math.tanh(u0)

        return u0

    def discriminator(
        self, inputs, targets, trainable=True, reuse=False, const_init=False
    ):
        # (n, 6, 256, 256)
        d0 = flow.concat([inputs, targets], axis=1)
        # (n, 64, 128, 128)
        d1 = self._downsample(
            d0,
            64,
            4,
            name="d_d1",
            apply_batchnorm=False,
            reuse=reuse,
            const_init=const_init,
            trainable=trainable,
        )
        # (n, 64, 64, 64)
        d2 = self._downsample(
            d1, 128, 4, name="d_d2", reuse=reuse, const_init=const_init
        )
        # (n, 256, 32, 32)
        d3 = self._downsample(
            d2, 256, 4, name="d_d3", reuse=reuse, const_init=const_init
        )
        # (n, 256, 34, 34)
        pad1 = flow.pad(d3, [[0, 0], [0, 0], [1, 1], [1, 1]])
        # (n, 512, 31, 31)
        conv1 = layers.conv2d(
            pad1,
            512,
            4,
            strides=1,
            padding="valid",
            name="d_conv1",
            trainable=trainable,
            reuse=reuse,
            const_init=const_init,
            use_bias=False,
        )
        bn1 = layers.batchnorm(conv1, name="d_bn", reuse=reuse, trainable=trainable)
        leaky_relu = flow.nn.leaky_relu(bn1, alpha=0.3)
        # (n, 512, 33, 33)
        pad2 = flow.pad(leaky_relu, [[0, 0], [0, 0], [1, 1], [1, 1]])
        # (n, 1, 30, 30)
        conv2 = layers.conv2d(
            pad2,
            1,
            4,
            strides=1,
            padding="valid",
            name="d_conv2",
            trainable=trainable,
            reuse=reuse,
            const_init=const_init,
        )

        return conv2

    def load_facades(self, data_path, mode="train"):
        from PIL import Image, ImageOps
        seed=np.random.randint(1024)
        if not os.path.exists(data_path):
            logger.info("not Found Facades - start download")
            import tensorflow as tf
            if not os.path.exists("data"):
                os.mkdir("data")
            _PATH = os.path.join(os.getcwd(), "data/facades.tar.gz")
            _URL = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz"
            path_to_zip = tf.keras.utils.get_file(_PATH, origin=_URL, extract=True)
        else:
            # logger.info("Found Facades - skip download")
            pass

        input_imgs, real_imgs = [], []
        if mode == "train":
            modes = ["train", "val"]
        else:
            modes = ["test"]

        for mode in modes:
            for d in os.listdir(os.path.join(data_path, mode)):
                d = os.path.join(data_path, mode, d)
                img = np.asarray(Image.open(d))
                real_img = Image.fromarray(img[:, :256, :])
                input_img = Image.fromarray(img[:, 256:, :])

                # resize to 286 x 286 x 3, and randomly crop to 256 x 256 x 3
                r1, r2 = np.random.randint(30, size=2)
                real_img = real_img.resize((256 + 30, 256 + 30))
                input_img = input_img.resize((256 + 30, 256 + 30))
                real_img = real_img.crop((r1, r2, r1 + 256, r2 + 256))
                input_img = input_img.crop((r1, r2, r1 + 256, r2 + 256))

                if np.random.rand() > 0.5:
                    # random mirroring
                    real_img = ImageOps.mirror(real_img)
                    input_img = ImageOps.mirror(input_img)
                
                real_imgs.append(np.asarray(real_img))
                input_imgs.append(np.asarray(input_img))

        input_imgs = np.array(input_imgs).transpose(0, 3, 1, 2)
        real_imgs = np.array(real_imgs).transpose(0, 3, 1, 2)
        # normalizing the images to [-1, 1]
        input_imgs = input_imgs / 127.5 - 1
        real_imgs = real_imgs / 127.5 - 1

        np.random.seed(seed)
        np.random.shuffle(input_imgs)
        np.random.seed(seed)
        np.random.shuffle(real_imgs)
        return input_imgs, real_imgs

    def save_images(self, images, real_input, target, epoch_idx, name, path=None):
        if name == "eval":
            plot_size = epoch_idx
        else:
            plot_size = self.batch_size

        if name == "train":
            images_path = self.train_images_path 
        elif name == "test":
            images_path = self.test_images_path

        plt.figure(figsize=(6, 8))
        display_list = list(zip(real_input, target, images))
        # title = ["Input Image", "Ground Truth", "Predicted Image"]
        idx = 1
        row = 4
        # save 4 images of title
        for i in range(plot_size):
            dis = display_list[i]
            for j in range(3):
                plt.subplot(row, 6, idx)
                # plt.title(title[j])
                # getting the pixel values between [0, 1] to plot it.
                plt.imshow(np.array(dis[j]).transpose(1, 2, 0) * 0.5 + 0.5)
                plt.axis("off")
                idx = idx + 1

            if idx > row * 6:
                break
        if name == "eval":
            save_path = path
        else:
            save_path = os.path.join(images_path, "{}_image_{:02d}.png".format(name, epoch_idx + 1))
        plt.savefig(save_path)
        plt.close()

    def test(self, eval_size, data_path, model_path, save_path):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        flow.config.gpu_device_num(self.gpus_per_node)
        @flow.global_function(type="predict", function_config=func_config)
        def eval_generator(
            input: tp.Numpy.Placeholder((eval_size, 3, 256, 256))
            ) -> tp.Numpy:
            g_out = self.generator(input, trainable=False)
            return g_out

        check_point = flow.train.CheckPoint()
        check_point.load(model_path)

        test_x, test_y = self.load_facades(data_path, mode="test")
        ind = np.random.choice(len(test_x) // eval_size)
        test_inp = test_x[ind * eval_size : (ind + 1) * eval_size].astype(np.float32, order="C")
        test_target = test_y[ind * eval_size : (ind + 1) * eval_size].astype(np.float32, order="C")
        gout = eval_generator(test_inp)
        # save images
        # self.save_images(gout, test_inp, test_target, eval_size, name="eval", path=save_path)

    def train(self, epochs, data_path, save=True):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        flow.config.gpu_device_num(self.gpus_per_node)
        lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [self.lr])

        @flow.global_function(type="train", function_config=func_config)
        def train_generator(
            input: tp.Numpy.Placeholder((self.batch_size, 3, 256, 256)),
            target: tp.Numpy.Placeholder((self.batch_size, 3, 256, 256)),
            label1: tp.Numpy.Placeholder((self.batch_size, 1, 30, 30)),
        )-> Tuple[tp.Numpy, tp.Numpy, tp.Numpy, tp.Numpy]:
            g_out = self.generator(input, trainable=True)
            g_logits = self.discriminator(input, g_out, trainable=False)
            gan_loss = flow.nn.sigmoid_cross_entropy_with_logits(
                label1, g_logits, name="Gloss_sigmoid_cross_entropy_with_logits"
            )
            l1_loss = flow.math.reduce_mean(flow.math.abs(g_out - target))
            g_loss = gan_loss + self.LAMBDA * l1_loss

            flow.optimizer.Adam(lr_scheduler, beta1=0.5).minimize(g_loss)
            return (g_out, gan_loss, l1_loss, g_loss)

        @flow.global_function(type="train", function_config=func_config)
        def train_discriminator(
            input: tp.Numpy.Placeholder((self.batch_size, 3, 256, 256)),
            target: tp.Numpy.Placeholder((self.batch_size, 3, 256, 256)),
            label0: tp.Numpy.Placeholder((self.batch_size, 1, 30, 30)),
            label1: tp.Numpy.Placeholder((self.batch_size, 1, 30, 30)),
        ) -> tp.Numpy:
            g_out = self.generator(input, trainable=False)
            g_logits = self.discriminator(g_out, target, trainable=True)
            d_fake_loss = flow.nn.sigmoid_cross_entropy_with_logits(
                label0, g_logits, name="Dloss_fake_sigmoid_cross_entropy_with_logits"
            )

            d_logits = self.discriminator(input, target, trainable=True, reuse=True)
            d_real_loss = flow.nn.sigmoid_cross_entropy_with_logits(
                label1, d_logits, name="Dloss_real_sigmoid_cross_entropy_with_logits"
            )

            d_loss = d_fake_loss + d_real_loss
            flow.optimizer.Adam(lr_scheduler, beta1=0.5).minimize(d_loss)

            return d_loss

        @flow.global_function(type="predict", function_config=func_config)
        def eval_generator(
            input: tp.Numpy.Placeholder((self.batch_size, 3, 256, 256)),
            target: tp.Numpy.Placeholder((self.batch_size, 3, 256, 256))
            ) -> Tuple[tp.Numpy, tp.Numpy]:
            g_out = self.generator(input, trainable=False)
            error = flow.math.reduce_mean(flow.math.abs(g_out - target))
            return (g_out, error)

        check_point = flow.train.CheckPoint()
        check_point.init()
        
        G_image_loss, G_GAN_loss, G_total_loss, D_loss = [], [], [], []
        test_G_image_error = []
        x, _ = self.load_facades(data_path)
        batch_num = len(x) // self.batch_size
        label1 = np.ones((self.batch_size, 1, 30, 30)).astype(np.float32)
        if self.label_smooth != 0:
            label1_smooth = label1 - self.label_smooth 
        label0 = np.zeros((self.batch_size, 1, 30, 30)).astype(np.float32)
     
        for epoch_idx in range(epochs):
            start = time.time()
            # run every epoch to shuffle
            x, y = self.load_facades(data_path)
            for batch_idx in range(batch_num):
                inp = x[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ].astype(np.float32, order="C")
                target = y[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ].astype(np.float32, order="C")
                # one_sided label smooth
                if self.label_smooth != 0:
                    d_loss = train_discriminator(inp, target, label0, label1_smooth)
                else:
                    d_loss = train_discriminator(inp, target, label0, label1)
                g_out, g_gan_loss, g_image_loss, g_total_loss = train_generator(inp, target, label1)
                g_gan_loss = g_gan_loss.mean()
                g_image_loss = g_image_loss.mean()
                g_total_loss = g_total_loss.mean()
                d_loss = d_loss.mean()

                G_GAN_loss.append(g_gan_loss)
                G_image_loss.append(g_image_loss)
                G_total_loss.append(g_total_loss)
                D_loss.append(d_loss)
                if (batch_idx + 1) % self.eval_interval == 0:
                    logger.info("############## train ###############")
                    logger.info(
                        "{}th epoch, {}th batch, dloss:{}, g_gan_loss:{}, g_image_loss:{}, g_total_loss:{}".format(
                            epoch_idx + 1, batch_idx + 1, d_loss, g_gan_loss, g_image_loss, g_total_loss 
                        )
                    )
            
            # calculate test error to validate the trained model
            if (epoch_idx + 1) % 20 == 0:
                # run every epoch to shuffle
                test_x, test_y = self.load_facades(mode="test")
                ind = np.random.choice(len(test_x) // self.batch_size)
                test_inp = test_x[ind * self.batch_size : (ind + 1) * self.batch_size].astype(np.float32, order="C")
                test_target = test_y[ind * self.batch_size : (ind + 1) * self.batch_size].astype(np.float32, order="C")
                gout, test_image_error = eval_generator(test_inp, test_target)
                # save images
                # self.save_images(g_out, inp, target, epoch_idx, name="train")
                # self.save_images(gout, test_inp, test_target, epoch_idx, name="test")
                logger.info("############## evaluation ###############")
                logger.info("{}th epoch, {}th batch, test_image_error:{}".format(epoch_idx + 1, batch_idx + 1, test_image_error.mean()))
            
            logger.info("Time for epoch {} is {} sec.".format(epoch_idx + 1, time.time() - start))

        if save:
            from datetime import datetime
            check_point.save(
                os.path.join(self.path, "pix2pix_{}".format(
                    str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))
                )
            )

        # save train loss and val error to plot
        # np.save(os.path.join(self.path, 'G_image_loss_{}.npy'.format(epochs)), G_image_loss)
        # np.save(os.path.join(self.path, 'G_GAN_loss_{}.npy'.format(epochs)), G_GAN_loss)
        # np.save(os.path.join(self.path, 'G_total_loss_{}.npy'.format(epochs)), G_total_loss)
        # np.save(os.path.join(self.path, 'D_loss_{}.npy'.format(epochs)), D_loss)
        logger.info("*************** Train {} done ***************** ".format(self.path))

    def save_to_gif(self):
        anim_file = os.path.join(self.path, "pix2pix .gif")
        with imageio.get_writer(anim_file, mode="I") as writer:
            filenames = glob.glob(os.path.join(self.test_images_path, "*image*.png"))
            filenames = sorted(filenames)
            last = -1
            for i, filename in enumerate(filenames):
                frame = 2 * (i ** 0.5)
                if round(frame) > round(last):
                    last = frame
                else:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)
        logger.info("Generate {} done.".format(anim_file))

if __name__ == "__main__":
    os.environ["ENABLE_USER_OP"] = "True"
    import argparse
    parser = argparse.ArgumentParser(description="flags for multi-node and resource")
    parser.add_argument("--data_url", type=str, default='./data/facades', required=True)
    parser.add_argument("--train_out", type=str, default='./', required=True, help="path of saving model")
    parser.add_argument("--train_log", type=str, default='./', required=True, help="path of saving model")
    parser.add_argument("--gpu_num_per_node", type=int, default=1, required=True)
    parser.add_argument("-e", "--epoch_num", type=int, default=200, required=False)
    parser.add_argument("-lr", "--learning_rate", type=float, default=2e-4, required=False)
    parser.add_argument("--LAMBDA", type=float, default=100, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--label_smooth", type=float, default=0, required=False)
    parser.add_argument("--test", action='store_true', default=False)
    args = parser.parse_args()
    
    args.train_log = os.path.join(args.train_log, "train_log")
    mkdirs(args.train_out, args.train_log)
    init_logger(args.train_log)
    logger.info(args)
    pix2pix = Pix2Pix(args)
    if not args.test:
        pix2pix.train(args.epoch_num, args.data_url)
        # if args.epoch_num > 20:
        #     pix2pix.save_to_gif()
    else:
        save_path = "eval_images.png"
        model_path = "./models/pix2pix"
        pix2pix.test(16, args.model_path, args.save_path, args.data_url)

