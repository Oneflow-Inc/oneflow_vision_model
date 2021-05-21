# import torch
# import torch.optim as optim
from utils.io import load_ckpt
from utils.io import save_ckpt
from modules.RFR import RFRNet
import os
import time
import oneflow as flow
import numpy as np
import oneflow.typing as tp
import cv2
import of_layers as layers
import datetime
import math
from typing import Tuple

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float32)
flow.config.enable_debug_mode(True)


# @flow.global_function(type="train",function_config=func_config)
# def train_job2(
#             maskedimg: tp.Numpy.Placeholder((6, 3, 256, 256), dtype=flow.float),
#             masks: tp.Numpy.Placeholder((6, 3, 256, 256), dtype=flow.float),
#             images: tp.Numpy.Placeholder((6, 3, 256, 256), dtype=flow.float),
# ):
#     init = RFRNetModel()
#     ret, mmask, fake, comp = init.buildnet(maskedimg, masks, images)
#     loss = init.get_g_loss(ret, mmask, fake, comp)
#     lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [5e-5])
#     # Set Adam optimizer
#     flow.optimizer.Adam(lr_scheduler, do_bias_correction=False).minimize(loss)
#     return loss

@flow.global_function(type="train", function_config=func_config)
def train_job(
        maskedimg: tp.Numpy.Placeholder((6, 3, 256, 256), dtype=flow.float),
        masks: tp.Numpy.Placeholder((6, 3, 256, 256), dtype=flow.float),
        images: tp.Numpy.Placeholder((6, 3, 256, 256), dtype=flow.float),
) -> tp.Numpy:
    init = RFRNetModel()
    ret, mmask, fake, comp = init.buildnet(maskedimg, masks, images)
    loss, l1_loss = init.get_g_loss(ret, mmask, fake, comp)
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [2e-4])
    # Set Adam optimizer
    flow.optimizer.Adam(lr_scheduler, do_bias_correction=False).minimize(loss)
    return l1_loss

@flow.global_function(type="predict", function_config=func_config)
def eval_job(images1: tp.Numpy.Placeholder((6, 3, 256, 256), dtype=flow.float),
             masks1: tp.Numpy.Placeholder((6, 3, 256, 256), dtype=flow.float),
             maskedimages1: tp.Numpy.Placeholder((6, 3, 256, 256), dtype=flow.float),
             ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy]:
    init = RFRNetModel()
    fake_B = init.buildnetest(maskedimages1, masks1)
    comp_B1 = fake_B * (1 - masks1) + images1 * masks1

    return comp_B1, masks1, maskedimages1


def train(train_loader, save_path, vgg_path, finetune=False, iters=600000, batch_size=6, epochs=10, ssiter=0):
    # writer = SummaryWriter(log_dir="log_info")
    # self.G.train(finetune = finetune)
    flow.load_variables(flow.checkpoint.get(vgg_path))
    # check_point = flow.train.CheckPoint()
    # check_point.load(vgg_path)
    # if finetune:
    # self.optm_G = optim.Adam(filter(lambda p:p.requires_grad, self.G.parameters()), lr = 5e-5)
    sum_l1 = 0.0
    batch_num = len(train_loader) // batch_size

    print("****************** start training *****************")
    for epoch_idx in range(epochs):
        train_loader.shuffle(epoch_idx)
        print("Starting training from iteration:{:d}".format(ssiter))
        s_time = time.time()
        while ssiter < iters:

            print("iters=", iters)
            print(len(train_loader))
            for batch_idx in range(batch_num):
                #print("ssiter", ssiter)
                imag = []
                maskk = []
                for idx in range(batch_size):
                    sample1 = train_loader[batch_idx * batch_size + idx]  # list输出是[2,256,256,3]
                    sample = np.array(sample1)
                    samp = np.transpose(sample, (0, 3, 1, 2))  # 已经是【2,3,256,256】
                    samp1 = samp[None, 0, :, :, :]
                    samp2 = samp[None, 1, :, :, :]
                    imag.append(samp1)
                    maskk.append(samp2)
                imag = np.ascontiguousarray(np.concatenate(imag, axis=0))
                imag2 = imag.astype(np.float32)
                maskk = np.ascontiguousarray(np.concatenate(maskk, axis=0))
                mask2 = maskk.astype(np.float32)
                # gt_images, masks = self.__cuda__(*items)
                masked_images = imag2 * mask2
                masked_image2 = masked_images.astype(np.float32)
                # self.forward(masked_images, masks, gt_images)
                # self.buildnet(masked_images, masks, gt_images) #global function
                # self.update_parameters()

                l1_loss_value = train_job(masked_image2, mask2, imag2)
                sum_l1 += l1_loss_value
                #print(l1_loss_value)
                # val_psnr += calc_psnr(sr, hr)
                # val_ssim += calc_ssim(sr, hr)
                ssiter += 1
                #
                if ssiter % 50 == 0:
                    e_time = time.time()
                    int_time = e_time - s_time
                    print("Epoch:%d,  Iteration:%d, l1_loss:%.4f, time_taken:%.2f" % (epoch_idx,ssiter, sum_l1 / 50, int_time))
                    s_time = time.time()
                    sum_l1 = 0.0
                if ssiter % 300000 == 0:
                    if not os.path.exists('{:s}'.format(save_path)):
                        os.makedirs('{:s}'.format(save_path))
                    save_ckpt('{:s}/epoch_{:d}_g_{:d}'.format(save_path,epoch_idx, ssiter))
        if not os.path.exists('{:s}'.format(save_path)):
            os.makedirs('{:s}'.format(save_path))
            save_ckpt('{:s}/epoch_{:d}_g_{:s}'.format(save_path,epoch_idx, "final"))


class RFRNetModel():
    def __init__(self):
        self.lossNet = None
        self.iter = None
        self.optm_G = None
        self.device = None
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.l1_loss_val = 0.0
        self.G = RFRNet()

    def initialize_model(self, path=None, train=True, it=0):
        # self.G = RFRNet()
        # self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-4)
        # if train:
        # self.lossNet = VGG16FeatureExtractor() #那在train里面加载vgg的参数就可以了
        try:
            load_ckpt(path)
            start_iter = it
            selfiter = start_iter
            if train:
                # self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-4)
                print('Model Initialized, iter: ', start_iter)

        except:
            print('No trained model, from start')
            selfiter = 0
        return selfiter

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
                trainable=True
        ):
            time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')
            name = name + str(time)
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
                inputs = conv_block[i]
                conv_i = conv2d_layer(
                    name="conv{}".format(index),
                    input=inputs,
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

        time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')
        results = [images]
        conv1 = _conv_block(images, 0, 64, 2, reuse=reuse, trainable=trainable)
        # pool1 = flow.nn.max_pool2d(conv1[-1], 2, 2, "VALID", "NCHW", name="pool1")
        pool1 = layers.max_pool2d(conv1[-1], 2, 2, name="pool1" + str(time), reuse=reuse)
        results.append(pool1)

        conv2 = _conv_block(pool1, 2, 128, 2, reuse=reuse, trainable=trainable)
        # pool2 = flow.nn.max_pool2d(conv2[-1], 2, 2, "VALID", "NCHW", name="pool2")
        pool2 = layers.max_pool2d(conv2[-1], 2, 2, name="pool2" + str(time), reuse=reuse)
        results.append(pool2)

        conv3 = _conv_block(pool2, 4, 256, 3, reuse=reuse, trainable=trainable)
        pool3 = layers.max_pool2d(conv3[-1], 2, 2, name="pool3" + str(time), reuse=reuse)
        results.append(pool3)
        return results[1:]
    def calc_psnr(self, img1, img2):
        ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
        diff = (img1 - img2)
        diff[:, :, 0] = diff[:, :, 0] * 25.738 / 256.0
        diff[:, :, 1] = diff[:, :, 1] * 25.057 / 256.0
        diff[:, :, 2] = diff[:, :, 2] * 25.064 / 256.0

        diff = np.sum(diff, axis=2)
        mse = np.mean(np.power(diff, 2))
        return -10 * math.log10(mse)
        #img1 = img1*255
        #img2 = img2*255
        #mse = np.mean( (img1 - img2) ** 2 )
        #mse = mse/3
        #if mse == 0:
            #return 100
        #PIXEL_MAX = 255.0
        #return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        
    def calc_ssim(self, img1, img2):
        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            img1 = img1*255
            img2 = img2*255
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

    def test(self, test_loader, result_save_path, batch_size=6):
        # self.G.eval()
        # for para in self.G.parameters():
        # para.requires_grad = False

        count = 0
        batch_num = len(test_loader) // batch_size

        print("****************** start testing *****************")

        # test_loader.shuffle(epoch_idx)

        print(len(test_loader))
        sum_ssim = 0.0
        sum_psnr = 0.0
        for batch_idx in range(batch_num):
            imag = []
            maskk = []
            for idx in range(batch_size):
                sample1 = test_loader[batch_idx * batch_size + idx]  # list输出是[2,256,256,3]
                sample = np.array(sample1)
                samp = np.transpose(sample, (0, 3, 1, 2))  # 已经是【2,3,256,256】
                samp1 = samp[None, 0, :, :, :]
                samp2 = samp[None, 1, :, :, :]
                imag.append(samp1)
                maskk.append(samp2)
            imag = np.ascontiguousarray(np.concatenate(imag, axis=0))
            imag2 = imag.astype(np.float32)
            maskk = np.ascontiguousarray(np.concatenate(maskk, axis=0))
            mask2 = maskk.astype(np.float32)
           
            maskediii = imag2 * mask2

            comp, maskkk, maskedimage2 = eval_job(imag2, mask2, maskediii)
            if not os.path.exists('{:s}/results'.format(result_save_path)):
                os.makedirs('{:s}/results'.format(result_save_path))

            sum1_ssim = 0.0
            sum1_psnr = 0.0
            for k in range(comp.shape[0]):
                count += 1
                # grid = make_grid(comp[k:k + 1])

                transpose_comp = np.transpose(comp[k],(1,2,0))
                transpose_masked = np.transpose(maskediii[k]+1 - mask2[k] ,(1,2,0))
                
                ssim_sum = self.calc_ssim(transpose_comp,transpose_masked)
                psnr_sum = self.calc_psnr(transpose_comp,transpose_masked)
                sum1_psnr+=psnr_sum
                sum1_ssim+=ssim_sum


                # file_path = '{:s}/results/img_{:d}.png'.format(result_save_path, count)
                # np.save(file_path, transpose_comp)


                # file_path = '{:s}/results/masked_img_{:d}.png'.format(result_save_path, count)
                # np.save(file_path, transpose_masked)
            sum2_ssim = sum1_ssim/comp.shape[0]
            sum2_psnr = sum1_psnr/comp.shape[0]
            sum_ssim += sum2_ssim
            sum_psnr += sum2_psnr
            if batch_idx % 100 ==0:
                print("The  batch is :", batch_idx)
                print("the  ssim is :", sum2_ssim)
                print("the  psnr is :", sum2_psnr)
        total_ssim = sum_ssim/batch_num
        total_psnr = sum_psnr/batch_num
        print("The all batch is :",batch_num)
        print("the avg of ssim :",total_ssim)
        print("the avg of psnr :",total_psnr)


    def buildnet(self, masked_image, mask, gt_image):
        # self.real_A = masked_image
        print("buildnet running")

        A = masked_image
        # self.real_B = gt_image
        B = gt_image
        # self.mask = mask
        real_mask = mask
        fake_B1 = self.G.buildnet(masked_image, mask)

        # self.fake_B = fake_B
        # self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
        comp_B1 = fake_B1 * (1 - mask) + B * mask

        return B, real_mask, fake_B1, comp_B1

    def buildnetest(self, masked_image, mask):

        # self.real_A = masked_image
        print("buildnetest running")

        A = masked_image

        real_mask = mask
        fake_B1 = self.G.buildnet(A, real_mask)

        # self.fake_B = fake_B
        # self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
        return fake_B1

    # def update_parameters(self):
    #     self.update_G()
    #     self.update_D()
    #
    # ef update_G(self):
    #     self.optm_G.zero_grad()
    #     loss_G = self.get_g_loss()
    #     loss_G.backward()
    #     self.optm_G.step()
    #
    # def update_D(self):
    #     return

    def get_g_loss(self, B, real_mask, fake_B1, comp_B1):
        real_B = B
        fake_B = fake_B1
        comp_B = comp_B1

        # real_B_feats = self.lossNet(real_B)
        real_B_feats = self.vgg16bn(real_B)
        real_B_feats = np.array(real_B_feats)
        # fake_B_feats = self.lossNet(fake_B)
        fake_B_feats = self.vgg16bn(fake_B)
        fake_B_feats = np.array(fake_B_feats)
        # comp_B_feats = self.lossNet(comp_B)
        comp_B_feats = self.vgg16bn(comp_B)
        comp_B_feats = np.array(comp_B_feats)
        print("vgg16 running")
        tv_loss = self.TV_loss(comp_B * (1 - real_mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats,
                                                                                                  comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, real_mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - real_mask))

        loss_G = (tv_loss * 0.1
                  + style_loss * 120
                  + preceptual_loss * 0.05
                  + valid_loss * 1
                  + hole_loss * 6)
        # self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        l1_loss_val = valid_loss + hole_loss
        return loss_G, l1_loss_val

    def l1_loss(self, f1, f2, mask=1):
        # return torch.mean(torch.abs(f1 - f2)*mask)
        return flow.math.reduce_mean(flow.math.abs(f1 - f2) * mask)

    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            # A_feat = flow.slice(x=A_feats,begin=[i,None,None,None,None],size=[1,None,None,None,None])
            # A_feat = flow.squeeze(input=A_feat,axis=[0])
            B_feat = B_feats[i]
            # B_feat = flow.slice(x=B_feats, begin=[i, None, None, None,None], size=[1, None, None, None,None])
            # B_feat = flow.squeeze(input=B_feat, axis=[0])
            # _, c, w, h = A_feat.size()
            _, c, w, h = A_feat.shape
            # A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            A_feat = flow.reshape(A_feat, [A_feat.shape[0], A_feat.shape[1], A_feat.shape[2] * A_feat.shape[3]])
            # B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            B_feat = flow.reshape(B_feat, [B_feat.shape[0], B_feat.shape[1], B_feat.shape[2] * B_feat.shape[3]])
            # A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            A_style = flow.matmul(A_feat, flow.transpose(A_feat, perm=[0, 2, 1]))
            # B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            B_style = flow.matmul(B_feat, flow.transpose(B_feat, perm=[0, 2, 1]))
            loss_value += flow.math.reduce_mean(flow.math.abs(A_style - B_style) / (c * w * h))
        return loss_value

    def TV_loss(self, x):
        # h_x = x.size(2)
        h_x = x.shape[2]
        # w_x = x.size(3)
        w_x = x.shape[3]
        k1 = flow.slice(x=x, begin=[None, None, 1, None], size=[None, None, h_x - 1, None])
        k1 = flow.squeeze(input=k1, axis=[2])
        k2 = flow.slice(x=x, begin=[None, None, 0, None], size=[None, None, h_x - 1, None])
        k2 = flow.squeeze(input=k2, axis=[2])
        h_tv = flow.math.reduce_mean(flow.math.abs(k1 - k2))
        k3 = flow.slice(x=x, begin=[None, None, None, 1], size=[None, None, None, w_x - 1])
        k3 = flow.squeeze(input=k3, axis=[3])
        k4 = flow.slice(x=x, begin=[None, None, None, 0], size=[None, None, None, w_x - 1])
        k4 = flow.squeeze(input=k4, axis=[3])
        w_tv = flow.math.reduce_mean(flow.math.abs(k3 - k4))
        return h_tv + w_tv

    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            # A_feat = flow.slice(x=A_feat, begin=[i, None, None, None], size=[1, None, None, None])
            # A_feat = flow.squeeze(input=A_feat, axis=[0])
            B_feat = B_feats[i]
            # B_feat = flow.slice(x=B_feat, begin=[i, None, None, None], size=[1, None, None, None])
            # B_feat = flow.squeeze(input=B_feat, axis=[0])
            loss_value += flow.math.reduce_mean(flow.math.abs(A_feat - B_feat))
        return loss_value



