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
import glob
import os
import test_layers as layers
import matplotlib.pyplot as plt
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class DCGAN:
    def __init__(self, args):
        self.lr = args.learning_rate
        self.z_dim = 100
        self.eval_interval = 50
        self.eval_size = 16
        self.seed = np.random.normal(0, 1, size=(self.eval_size, self.z_dim)).astype(
            np.float32)
        self.path = args.path
        if not os.path.exists(self.path):
            os.mkdir(self.path)
            print("Make new dir '{}' done.".format(self.path))
        self.gpus_per_node = args.gpu_num_per_node
        self.label_smooth = args.label_smooth
        self.G_loss = []
        self.D_loss = []

        self.batch_size = args.batch_size * self.gpus_per_node

        self.checkpoint_path = os.path.join(self.path, "checkpoint")
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.images_path = os.path.join(self.path, "images")
        if not os.path.exists(self.images_path):
            os.mkdir(self.images_path)
        # self.train_images_path = os.path.join(self.path, "train_images")
        # if not os.path.exists(self.train_images_path):
        #     os.mkdir(self.train_images_path)

    def train(self, epochs=1, model_dir=None, save=True):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        flow.config.gpu_device_num(self.gpus_per_node)
        lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [self.lr])

        @flow.global_function(type="train", function_config=func_config)
        def train_generator(
            z: tp.Numpy.Placeholder((self.batch_size, self.z_dim)),
            label1: tp.Numpy.Placeholder((self.batch_size, 1))
        ) -> Tuple[tp.Numpy, tp.Numpy]:
            g_out = self.generator(z, trainable=True)
            g_logits = self.discriminator(g_out, trainable=False)
            g_loss = flow.nn.sigmoid_cross_entropy_with_logits(
                label1, g_logits, name="Gloss_sigmoid_cross_entropy_with_logits"
            )
            g_loss = flow.math.reduce_mean(g_loss)
            
            flow.optimizer.Adam(lr_scheduler).minimize(g_loss)
        
            return (g_loss, g_out)

        @flow.global_function(type="train", function_config=func_config)
        def train_discriminator(
            z: tp.Numpy.Placeholder((self.batch_size, 100)),
            images: tp.Numpy.Placeholder((self.batch_size, 1, 28, 28)),
            label1: tp.Numpy.Placeholder((self.batch_size, 1)),
            label0: tp.Numpy.Placeholder((self.batch_size, 1))
        )-> Tuple[tp.Numpy, tp.Numpy, tp.Numpy]:
            g_out = self.generator(z, trainable=False)
            g_logits = self.discriminator(g_out, trainable=True)
            d_loss_fake = flow.nn.sigmoid_cross_entropy_with_logits(
                label0, g_logits, name="Dloss_fake_sigmoid_cross_entropy_with_logits"
            )

            d_logits = self.discriminator(images, trainable=True, reuse=True)
            
            d_loss_real = flow.nn.sigmoid_cross_entropy_with_logits(
                label1, d_logits, name="Dloss_real_sigmoid_cross_entropy_with_logits"
            )
            d_loss = d_loss_fake + d_loss_real

            flow.optimizer.Adam(lr_scheduler).minimize(d_loss)

            return (d_loss, d_loss_fake, d_loss_real)

        @flow.global_function(type="predict", function_config=func_config)
        def eval_generator(
            z: tp.Numpy.Placeholder((self.eval_size, self.z_dim))
            )-> tp.Numpy:
            g_out = self.generator(z, trainable=False)
            return g_out

        check_point = flow.train.CheckPoint()
        check_point.init()

        x, _ = self.load_mnist()
        batch_num = len(x) // self.batch_size
        label1 = np.ones((self.batch_size, 1)).astype(np.float32)
        label0 = np.zeros((self.batch_size, 1)).astype(np.float32)
        if self.label_smooth != 0:
            label1_smooth = label1 - self.label_smooth 
        for epoch_idx in range(epochs):
            start = time.time()
            for batch_idx in range(batch_num):
                z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim)).astype(np.float32)
                images = x[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ].astype(np.float32)
                if self.label_smooth != 0:
                    d_loss, d_loss_fake, d_loss_real = train_discriminator(z, images, label1_smooth, label0)
                else:
                    d_loss, d_loss_fake, d_loss_real = train_discriminator(z, images, label1, label0)
                g_loss, g_out = train_generator(z, label1)

                if (batch_idx + 1) % 10 == 0:
                    self.G_loss.append(g_loss.mean())
                    self.D_loss.append(d_loss.mean())
                
                batch_total = batch_idx + epoch_idx * batch_num * self.batch_size
                if (batch_idx + 1) % self.eval_interval == 0:
                    print(
                        "{}th epoch, {}th batch, d_fakeloss:{:>12.10f}, d_realloss:{:>12.10f}, d_loss:{:>12.10f}, g_loss:{:>12.10f}".format(
                            epoch_idx + 1, batch_idx + 1, d_loss_fake.mean(), d_loss_real.mean(), d_loss.mean(), g_loss.mean()
                        )
                    )

            self._eval_model_and_save_images(
                eval_generator, batch_idx + 1, epoch_idx + 1
            )
            # self.save_images(g_out, batch_idx + 1, epoch_idx + 1
            # )
            print("Time for epoch {} is {} sec.".format(epoch_idx + 1, time.time() - start))

        if save:
            from datetime import datetime
            check_point.save(
                os.path.join(self.checkpoint_path, "dcgan_{}".format(
                    str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))
                )
            )
        
        np.save(os.path.join(self.path, 'g_loss_{}.npy'.format(epochs)), self.G_loss)
        np.save(os.path.join(self.path, 'd_loss_{}.npy'.format(epochs)), self.D_loss)

    def test(self, eval_size, save_path, model_path):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        @flow.global_function(type="predict", function_config=func_config)
        def eval_generator(
            z: tp.Numpy.Placeholder((eval_size, self.z_dim))
            ) -> tp.Numpy:
            g_out = self.generator(z, trainable=False)
            return g_out

        check_point = flow.train.CheckPoint()
        check_point.load(model_path)
        z = np.random.normal(0, 1, size=(eval_size, self.z_dim)).astype(np.float32)
        results = eval_generator(z)
        self.save_images(results, eval_size, 0, "test", save_path)

    def save_to_gif(self):
        anim_file = os.path.join(self.path, "dcgan.gif")
        with imageio.get_writer(anim_file, mode="I") as writer:
            filenames = glob.glob(os.path.join(self.images_path, "*image*.png"))
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
        print("Generate {} done.".format(anim_file))

    def _eval_model_and_save_images(self, model, batch_idx, epoch_idx):
        results = model(self.seed)
        fig = plt.figure(figsize=(4, 4))
        for i in range(self.eval_size):
            plt.subplot(4, 4, i + 1)
            plt.imshow(results[i, 0, :, :] * 127.5 + 127.5, cmap="gray")
            plt.axis("off")
        plt.savefig(os.path.join(self.images_path, "image_{:02d}_{:04d}.png".format(epoch_idx, batch_idx)))

    def save_images(self, images, batch_idx, epoch_idx, mode="train", save_path=None):
        results = images
        fig = plt.figure(figsize=(4, 4))
        if mode == "train":
            for i in range(self.eval_size):
                plt.subplot(4, 4, i + 1)
                plt.imshow(results[i, 0, :, :] * 127.5 + 127.5, cmap="gray")
                plt.axis("off")
            plt.savefig(os.path.join(self.train_images_path, "train_image_{:02d}_{:04d}.png".format(epoch_idx, batch_idx)))
        else:
            # test, batch_idx = eval_size
            for i in range(batch_idx):
                plt.subplot(4, 4, i + 1)
                plt.imshow(results[i, 0, :, :] * 127.5 + 127.5, cmap="gray")
                plt.axis("off")
            plt.savefig(save_path)

    def generator(self, z, const_init=False, trainable=True):
        # (n, 256, 7, 7)
        h0 = layers.dense(
            z, 7 * 7 * 256, name="g_fc1", const_init=const_init, trainable=trainable
        )
        h0 = layers.batchnorm(h0, axis=1, name="g_bn1", trainable=trainable)
        # h0 = layers.batchnorm(h0, axis=1, name="g_bn1")
        h0 = flow.nn.leaky_relu(h0, 0.3)
        h0 = flow.reshape(h0, (-1, 256, 7, 7))
        # (n, 128, 7, 7)
        h1 = layers.deconv2d(
            h0,
            128,
            5,
            strides=1,
            name="g_deconv1",
            const_init=const_init,
            trainable=trainable,
        )
        h1 = layers.batchnorm(h1, name="g_bn2", trainable=trainable)
        # h1 = layers.batchnorm(h1, name="g_bn2")
        h1 = flow.nn.leaky_relu(h1, 0.3)
        # (n, 64, 14, 14)
        h2 = layers.deconv2d(
            h1,
            64,
            5,
            strides=2,
            name="g_deconv2",
            const_init=const_init,
            trainable=trainable,
        )
        h2 = layers.batchnorm(h2, name="g_bn3", trainable=trainable)
        # h2 = layers.batchnorm(h2, name="g_bn3")
        h2 = flow.nn.leaky_relu(h2, 0.3)
        # (n, 1, 28, 28)
        out = layers.deconv2d(
            h2,
            1,
            5,
            strides=2,
            name="g_deconv3",
            const_init=const_init,
            trainable=trainable,
        )
        out = flow.math.tanh(out)
        return out

    def discriminator(self, img, const_init=False, trainable=True, reuse=False):
        # (n, 1, 28, 28)
        h0 = layers.conv2d(
            img,
            64,
            5,
            name="d_conv1",
            const_init=const_init,
            trainable=trainable,
            reuse=reuse,
        )
        h0 = flow.nn.leaky_relu(h0, 0.3)
        h0 = flow.nn.dropout(h0, rate=0.3)
        # (n, 64, 14, 14)
        h1 = layers.conv2d(
            h0,
            128,
            5,
            name="d_conv2",
            const_init=const_init,
            trainable=trainable,
            reuse=reuse,
        )
        h1 = flow.nn.leaky_relu(h1, 0.3)
        h1 = flow.nn.dropout(h1, rate=0.3)
        # (n, 128 * 7 * 7)
        out = flow.reshape(h1, (self.batch_size, -1))
        # (n, 1)
        out = layers.dense(
            out, 1, name="d_fc", const_init=const_init, trainable=trainable, reuse=reuse
        )
        return out

    def download_mnist(self, data_dir):
        import subprocess
        os.mkdir(data_dir)
        url_base = "http://yann.lecun.com/exdb/mnist/"
        file_names = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ]
        for file_name in file_names:
            url = (url_base + file_name).format(**locals())
            print(url)
            out_path = os.path.join(data_dir, file_name)
            cmd = ["curl", url, "-o", out_path]
            print("Downloading ", file_name)
            subprocess.call(cmd)
            cmd = ["gzip", "-d", out_path]
            print("Decompressing ", file_name)
            subprocess.call(cmd)

    def load_mnist(self, root_dir="./data", dataset_name="mnist", transpose=True):
        data_dir = os.path.join(root_dir, dataset_name)
        if os.path.exists(data_dir):
            print("Found MNIST - skip download")
        else:
            print("not Found MNIST - start download")
            if not os.path.exists(root_dir):
                os.mkdir(root_dir)
            self.download_mnist(data_dir)

        fd = open(os.path.join(data_dir, "train-images-idx3-ubyte"))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, "train-labels-idx1-ubyte"))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, "t10k-images-idx3-ubyte"))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = trX
        y = trY.astype(np.int)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), 10), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        if transpose:
            X = np.transpose(X, (0, 3, 1, 2))

        # return X / 255.0, y_vec
        return (X - 127.5) / 127.5, y_vec


if __name__ == "__main__":
    os.environ["ENABLE_USER_OP"] = "True"
    import argparse
    parser = argparse.ArgumentParser(description="flags for multi-node and resource")
    parser.add_argument("--path", type=str, default='of_model', required=False)
    parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
    parser.add_argument("--epoch_num", type=int, default=100, required=False)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, required=False)
    parser.add_argument(
        "-load", "--model_load_dir", type=str, default="./training_checkpoints/checkpoint", required=False
    )
    parser.add_argument("--batch_size", type=int, default=200, required=False)
    parser.add_argument("--label_smooth", type=float, default=0.15, required=False)
    parser.add_argument("--test", action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    dcgan = DCGAN(args)
    if not args.test:
        # train 
        dcgan.train(args.epoch_num)
        dcgan.save_to_gif()
    else:
        # test
        save_path = "eval_images.png"
        model_path = "./models/dcgan"
        dcgan.test(16, save_path, model_path)