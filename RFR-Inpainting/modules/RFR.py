# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from modules.partialconv import PartialConv2d
from modules.Attent import AttentionModule
# from torchvision import models
import numpy as np
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple
import datetime


# class VGG16FeatureExtractor(object):   #vgg16特征提取
# def __init__(self):
# super().__init__()
# vgg16 = models.vgg16(pretrained=True)
# self.enc_1 = nn.Sequential(*vgg16.features[:5])
# self.enc_2 = nn.Sequential(*vgg16.features[5:10])
# self.enc_3 = nn.Sequential(*vgg16.features[10:17])

# fix the encoder
# for i in range(3):
# for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
# param.requires_grad = False

# def buildnet(self, image):
# results = [image]
# for i in range(3):
# func = getattr(self, 'enc_{:d}'.format(i + 1))


# results.append(func(results[-1]))
# return results[1:]


class Bottleneck(object):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,time=""):
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.time =time
    def buildnet(self, x):
        residual = x
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # out = self.conv1(x)
        out = flow.layers.conv2d(x, self.planes, kernel_size=1, use_bias=False, name="conv1"+self.time)
        # self.bn1 = nn.BatchNorm2d(planes)
        # out = self.bn1(out)
        out = flow.layers.batch_normalization(inputs=out,
                                              axis=1,
                                              momentum=0.997,
                                              epsilon=1.001e-5,
                                              center=True,
                                              scale=True,
                                              trainable=True,
                                              name="bn1"+self.time)
        # self.relu = nn.ReLU(inplace=True)
        # out = self.relu(out)
        out = flow.nn.relu(out)

        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        # padding=1, bias=False)
        # out = self.conv2(out)
        out = flow.layers.conv2d(out, self.planes, kernel_size=3, use_bias=False, strides=self.stride,
                                 padding=[0, 0, 1, 1], name="conv2"+self.time)
        # self.bn2 = nn.BatchNorm2d(planes)
        # out = self.bn2(out)
        out = flow.layers.batch_normalization(inputs=out,
                                              axis=1,
                                              momentum=0.997,
                                              epsilon=1.001e-5,
                                              center=True,
                                              scale=True,
                                              trainable=True,
                                              name="bn2"+self.time)
        # self.relu = nn.ReLU(inplace=True)
        # out = self.relu(out)
        out = flow.nn.relu(out)

        # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # out = self.conv3(out)
        out = flow.layers.conv2d(out, self.planes * 4, kernel_size=1, use_bias=False, name="conv3"+self.time)
        # self.bn3 = nn.BatchNorm2d(planes * 4)
        # out = self.bn3(out)
        out = flow.layers.batch_normalization(inputs=out,
                                              axis=1,
                                              momentum=0.997,
                                              epsilon=1.001e-5,
                                              center=True,
                                              scale=True,
                                              trainable=True,
                                              name="bn3"+self.time)

        out = flow.math.add(residual, out)
        # self.relu = nn.ReLU(inplace=True)
        # out = self.relu(out)
        out = flow.nn.relu(out)
        return out


class RFRModule(object):
    def __init__(self, layer_size=6, in_channel=64,time=""):
        super(RFRModule, self).__init__()
        self.freeze_enc_bn = False
        self.layer_size = layer_size
        self.in_channel = in_channel
        self.time =time
        self.att = AttentionModule(512, tim="at" + time)
        # for i in range(3):
        # name = 'enc_{:d}'.format(i + 1)   #name = 'enc_1'
        # out_channel = in_channel * 2
        # block = [nn.Conv2d(in_channel, out_channel, 3, 2, 1, bias = False),
        # nn.BatchNorm2d(out_channel),
        # nn.ReLU(inplace = True)]
        # in_channel = out_channel
        # setattr(self, name, nn.Sequential(*block))    #self.enc_i=nn.sequential(*block)

        # for i in range(3, 6):
        # name = 'enc_{:d}'.format(i + 1)
        # block = [nn.Conv2d(in_channel, out_channel, 3, 1, 2, dilation = 2, bias = False),
        # nn.BatchNorm2d(out_channel),
        # nn.ReLU(inplace = True)]
        # setattr(self, name, nn.Sequential(*block))
        # self.att = AttentionModule(512)
        # for i in range(5, 3, -1):
        # name = 'dec_{:d}'.format(i)
        # block = [nn.Conv2d(in_channel + in_channel, in_channel, 3, 1, 2, dilation = 2, bias = False),
        # nn.BatchNorm2d(in_channel),
        # nn.LeakyReLU(0.2, inplace = True)]
        # setattr(self, name, nn.Sequential(*block))

        # block = [nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias = False),
        # nn.BatchNorm2d(512),
        # nn.LeakyReLU(0.2, inplace = True)]
        # self.dec_3 = nn.Sequential(*block)

        # block = [nn.ConvTranspose2d(768, 256, 4, 2, 1, bias = False),
        # nn.BatchNorm2d(256),
        # nn.LeakyReLU(0.2, inplace = True)]
        # self.dec_2 = nn.Sequential(*block)

        # block = [nn.ConvTranspose2d(384, 64, 4, 2, 1, bias = False),
        # nn.BatchNorm2d(64),
        # nn.LeakyReLU(0.2, inplace = True)]
        # self.dec_1 = nn.Sequential(*block)

    def buildnet(self, input, mask):
        #time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')
        # inch = self.in_channel
        h_dict = {}  # for the output of enc_N

        h_dict['h_0'] = input

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):  # 1-7
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            # h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])
            # h_dict['h_1']=self.enc_1(h_dict['h_0'])==self.enc_1(input) == 获得对象的某个属性值

            if i >= 1 and i <= 3:  # 1
                self.out_channel = self.in_channel * 2
                getit = flow.layers.conv2d(inputs=h_dict[h_key_prev], filters=self.out_channel, kernel_size=[3, 3],
                                           strides=2, padding=[0, 0, 1, 1], use_bias=False,
                                           name="enconv" + str(i) + self.time)
                getit = flow.layers.batch_normalization(inputs=getit,
                                                        axis=1,
                                                        momentum=0.997,
                                                        epsilon=1.001e-5,
                                                        center=True,
                                                        scale=True,
                                                        trainable=True,
                                                        name="bn11" + str(i) + self.time)
                h_dict[h_key] = flow.nn.relu(getit)
                self.in_channel = self.out_channel

            if i > 3 and i <= 6:  # 2
                self.out_channel = self.in_channel
                getit = flow.layers.conv2d(inputs=h_dict[h_key_prev], filters=self.out_channel, kernel_size=[3, 3],
                                           strides=1, padding=[0, 0, 2, 2], use_bias=False, dilation_rate=2,
                                           name="enconv" + str(i) + self.time)
                getit = flow.layers.batch_normalization(inputs=getit,
                                                        axis=1,
                                                        momentum=0.997,
                                                        epsilon=1.001e-5,
                                                        center=True,
                                                        scale=True,
                                                        trainable=True,
                                                        name="bn12" + str(i) + self.time)
                h_dict[h_key] = flow.nn.relu(getit)

            h_key_prev = h_key

        h = h_dict[h_key]
        inch = self.out_channel
        for i in range(self.layer_size - 1, 0, -1):  # 5-1
            enc_h_key = 'h_{:d}'.format(i)
            dec_l_key = 'dec_{:d}'.format(i)
            # h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h = flow.concat(inputs=[h, h_dict[enc_h_key]], axis=1)
            # h = getattr(self, dec_l_key)(h) self.dec_i = h
            if i >= 4 and i <= 5:
                getit = flow.layers.conv2d(inputs=h, filters=inch, kernel_size=[3, 3],
                                           strides=1, padding=[0, 0, 2, 2], use_bias=False, dilation_rate=2,
                                           name="deconv" + str(i) + self.time)
                getit = flow.layers.batch_normalization(inputs=getit,
                                                        axis=1,
                                                        momentum=0.997,
                                                        epsilon=1.001e-5,
                                                        center=True,
                                                        scale=True,
                                                        trainable=True,
                                                        name="bn13" + str(i) + self.time)
                h = flow.nn.leaky_relu(x=getit, alpha=0.2)

            if i == 3:
                init3 = flow.random_normal_initializer(mean=0.0, stddev=0.02)
                weight_regularizer = flow.regularizers.l2(0.0005)
                weight_shape1 = (1024,
                                 512,
                                 4,
                                 4)
                weight = flow.get_variable(
                    "-weight3",
                    shape=weight_shape1,
                    initializer=init3,
                    regularizer=weight_regularizer,
                )
                getit3 = flow.nn.torch_conv2d_transpose(value=h, filter=weight, padding_needed=(2, 2), strides=2,
                                                        name="deconv3" + self.time, output_padding=(0, 0))
                getit3 = flow.layers.batch_normalization(inputs=getit3,
                                                         axis=1,
                                                         momentum=0.997,
                                                         epsilon=1.001e-5,
                                                         center=True,
                                                         scale=True,
                                                         trainable=True,
                                                         name="bn14" + self.time)
                h = flow.nn.leaky_relu(x=getit3, alpha=0.2)
                # self.att = AttentionModule(512)
                h = self.att.buildnet(h, mask)

            if i == 2:
                init2 = flow.random_normal_initializer(mean=0.0, stddev=0.02)
                weight_regularizer = flow.regularizers.l2(0.0005)
                weight_shape2 = (768,
                                 256,
                                 4,
                                 4)
                weight = flow.get_variable(
                    "-weight2",
                    shape=weight_shape2,
                    initializer=init2,
                    regularizer=weight_regularizer,
                )
                getit2 = flow.nn.torch_conv2d_transpose(value=h, filter=weight, padding_needed=(2, 2), strides=2,
                                                        name="deconv2" + self.time, output_padding=(0, 0))
                getit2 = flow.layers.batch_normalization(inputs=getit2,
                                                         axis=1,
                                                         momentum=0.997,
                                                         epsilon=1.001e-5,
                                                         center=True,
                                                         scale=True,
                                                         trainable=True,
                                                         name="bn15" + self.time)
                h = flow.nn.leaky_relu(x=getit2, alpha=0.2)
            if i == 1:
                init1 = flow.random_normal_initializer(mean=0.0, stddev=0.02)
                weight_regularizer = flow.regularizers.l2(0.0005)
                weight_shape3 = (384,
                                 64,
                                 4,
                                 4)
                weight = flow.get_variable(
                    "-weight1",
                    shape=weight_shape3,
                    initializer=init1,
                    regularizer=weight_regularizer,
                )
                getit1 = flow.nn.torch_conv2d_transpose(value=h, filter=weight, padding_needed=(2, 2), strides=2,
                                                        name="deconv1" + self.time, output_padding=(0, 0))
                getit1 = flow.layers.batch_normalization(inputs=getit1,
                                                         axis=1,
                                                         momentum=0.997,
                                                         epsilon=1.001e-5,
                                                         center=True,
                                                         scale=True,
                                                         trainable=True,
                                                         name="bn16" +self.time)
                h = flow.nn.leaky_relu(x=getit1, alpha=0.2)
        return h


class RFRNet(object):
    def __init__(self):
        super(RFRNet, self).__init__()
        # self.Pconv1 = PartialConv2d(3, 64, 7, 2, 3, multi_channel = True, bias = False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.Pconv2 = PartialConv2d(64, 64, 7, 1, 3, multi_channel = True, bias = False)
        # self.bn20 = nn.BatchNorm2d(64)
        # self.Pconv21 = PartialConv2d(64, 64, 7, 1, 3, multi_channel = True, bias = False)
        # self.Pconv22 = PartialConv2d(64, 64, 7, 1, 3, multi_channel = True, bias = False)
        # self.bn2 = nn.BatchNorm2d(64)

        # self.Tconv = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias = False)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.tail1 = PartialConv2d(67, 32, 3, 1, 1, multi_channel = True, bias = False)
        # self.tail2 = Bottleneck(32,8)
        # self.out = nn.Conv2d(64,3,3,1,1, bias = False)

    def buildnet(self, in_image, mask):
        #time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')
        self.Pconv1 = PartialConv2d(3, 64, 7, 2, 3, multi_channel=True, bias=False,time="part1")
        x1, m1 = self.Pconv1.buildnet(in_image, mask)
        # self.bn1 = nn.BatchNorm2d(64)
        # x1 = F.relu(self.bn1(x1), inplace = True)
        output = flow.layers.batch_normalization(inputs=x1, axis=1, momentum=0.997, epsilon=1.001e-5, center=True,
                                                 scale=True, trainable=True, name="_bn")
        x1 = flow.nn.relu(output)
        self.Pconv2 = PartialConv2d(64, 64, 7, 1, 3, multi_channel=True, bias=False,time="part2")
        x1, m1 = self.Pconv2.buildnet(x1, m1)
        # self.bn20 = nn.BatchNorm2d(64)
        # x1 = F.relu(self.bn20(x1), inplace = True)
        output1 = flow.layers.batch_normalization(inputs=x1, axis=1, momentum=0.997, epsilon=1.001e-5, center=True,
                                                  scale=True, trainable=True, name="_bn2")
        x1 = flow.nn.relu(output1)
        x2 = x1
        x2, m2 = x1, m1
        n, c, h, w = x2.shape
        # feature_group = [x2.view(n, c, 1, h, w)]
        feature_group = [flow.reshape(x=x2, shape=(n, c, 1, h, w))]
        # mask_group = [m2.view(n, c, 1, h, w)]
        mask_group = [flow.reshape(x=m2, shape=(n, c, 1, h, w))]
        self.RFRModule = RFRModule(time="RFRMo1")
        art = self.RFRModule.att
        art1 = art.attt
        art1.att_scores_prev = None
        art1.masks_prev = None

        for i in range(6):
            #time1 = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')
            self.Pconv21 = PartialConv2d(64, 64, 7, 1, 3, multi_channel=True, bias=False,time="part3"+str(i))
            self.Pconv22 = PartialConv2d(64, 64, 7, 1, 3, multi_channel=True, bias=False,time="part4"+str(i))
            x2, m2 = self.Pconv21.buildnet(x2, m2)
            x2, m2 = self.Pconv22.buildnet(x2, m2)
            # self.bn2 = nn.BatchNorm2d(64)
            output2 = flow.layers.batch_normalization(inputs=x2, axis=1, momentum=0.997, epsilon=1.001e-5, center=True,
                                                      scale=True, trainable=True, name="_bn3" + str(i))
            x2 = flow.nn.leaky_relu(output2)
            # x2 = F.leaky_relu(self.bn2(x2), inplace = True)
            model1 = RFRModule(time="RFRmo2"+str(i))
            # x2 = model1(x2, m2[:,0:1,:,:])

            x2 = model1.buildnet(x2, flow.slice(x=m2, begin=[None, 0, None, None], size=[None, 1, None, None]))
            x2 = x2 * m2
            feature_group.append(flow.reshape(x=x2, shape=(n, c, 1, h, w)))
            mask_group.append(flow.reshape(x=m2, shape=(n, c, 1, h, w)))
        # x3 = torch.cat(feature_group, dim = 2)
        x3 = flow.concat(inputs=feature_group, axis=2)
        # m3 = torch.cat(mask_group, dim = 2)
        m3 = flow.concat(inputs=mask_group, axis=2)
        # amp_vec = m3.mean(dim = 2)
        amp_vec = flow.math.reduce_mean(m3, axis=2)
        # x3 = (x3*m3).mean(dim = 2) /(amp_vec+1e-7)
        x3 = flow.math.reduce_mean(x3 * m3, axis=2) / (amp_vec + 1e-7)
        # x3 = x3.view(n, c, h, w)
        x3 = flow.reshape(x=x3, shape=(n, c, h, w))
        # m3 = m3[:,:,-1,:,:]
        m3 = flow.slice(x=m3, begin=[None, None, -2, None, None], size=[None, None, 1, None, None])
        m3 = flow.squeeze(input=m3, axis=[2])
        # self.Tconv = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False)
        # x4 = self.Tconv(x3)
        init4 = flow.random_normal_initializer(mean=0.0, stddev=0.02)
        weight_regularizer = flow.regularizers.l2(0.0005)
        weight = flow.get_variable(
            "-weight111",
            shape=(64, 64, 4, 4),
            initializer=init4,
            regularizer=weight_regularizer,
        )
        x4 = flow.nn.torch_conv2d_transpose(value=x3, filter=weight, padding_needed=(2, 2), strides=2,
                                            name="deconv1" , output_padding=(0, 0))
        # self.bn3 = nn.BatchNorm2d(64)
        # x4 = F.leaky_relu(self.bn3(x4), inplace = True)
        output3 = flow.layers.batch_normalization(inputs=x4, axis=1, momentum=0.997, epsilon=1.001e-5, center=True,
                                                  scale=True, trainable=True, name="output_bn3" )
        x4 = flow.nn.leaky_relu(output3)

        # m4 = F.interpolate(m3, scale_factor = 2)
        m4 = flow.layers.upsample_2d(x=m3, size=(2, 2))
        # x5 = torch.cat([in_image, x4], dim = 1)
        x5 = flow.concat(inputs=[in_image, x4], axis=1)
        # m5 = torch.cat([mask, m4], dim = 1)
        m5 = flow.concat(inputs=[mask, m4], axis=1)
        self.tail1 = PartialConv2d(67, 32, 3, 1, 1, multi_channel=True, bias=False,time="part5")
        x5, _ = self.tail1.buildnet(x5, m5)
        # x5 = F.leaky_relu(x5, inplace = True)
        x5 = flow.nn.leaky_relu(x5)
        self.tail2 = Bottleneck(32, 8,time="bott")
        x6 = self.tail2.buildnet(x5)
        # x6 = torch.cat([x5,x6], dim = 1)
        x6 = flow.concat(inputs=[x5, x6], axis=1)
        # self.out = nn.Conv2d(64, 3, 3, 1, 1, bias=False)
        # output = self.out(x6)
        init0 = flow.random_normal_initializer(mean=0.0, stddev=0.02)
        output = flow.layers.conv2d(inputs=x6, filters=3, kernel_size=3, strides=1, padding=[0, 0, 1, 1],
                                    kernel_initializer=init0,
                                    name="conv31")
        return output

    def train(self, mode=True, finetune=False):
        super().train(mode)
        if finetune:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()




# input =torch.randn(1,64,128,128)
# filter =torch.randn(1,1,32,32)
# conv = RFRNet()
# output,_=conv(input,filter)
# print(output.size())