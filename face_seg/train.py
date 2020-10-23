"""
# Version:0.0.1
# Date:14/10/2020
# Author: Jiaojiao Ye (jiaojiaoye@zhejianglab.com), Jiaqin Fu (jiaqinfu@zhejianglab.com)
"""

import oneflow as flow
import oneflow.typing as tp
import numpy as np
import matplotlib.pyplot as plt
import os

# custome library
from model import LinkNet34
from eval import * 
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = ' 0 '

def parse_args():
    parser = argparse.ArgumentParser(description='Face segmentation')
    parser.add_argument("--epoch", type=int, default=40, required=False)
    # for oneflow
    parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
    parser.add_argument("--lr", type=int, default=0.01, required=False, help="learning rate")
    parser.add_argument("--model_load_dir", type=str, default='./faceseg_backbones', required=False, help="model load directory")
    parser.add_argument("--dataset_dir", type=str, default='./data/', required=False, help="dataset root directory")
    parser.add_argument("--log_dir", type=str, default="./output", required=False, help="log info save directory")
    parser.add_argument("--out_dir", type=str, default="./output/save_model/", required=False)
    parser.add_argument("--img_height", type=int, default=256, required=False)
    parser.add_argument("--img_width", type=int, default=256, required=False)
    parser.add_argument("--train_batch_size", type=int, default=128, required=False)
    parser.add_argument("--val_batch_size", type=int, default=128, required=False)
    parser.add_argument("--jaccard_weight", type=float, default=0.5, required=False,  help='jaccard weight for loss, a float between 0 and 1.')

    args = parser.parse_args()
    return args


# train config
args = parse_args()
func_config = flow.function_config()
func_config.default_data_type(flow.float)
flow.config.gpu_device_num(args.gpu_num_per_node)

# hyperparameters
batch_size = args.train_batch_size
val_batch_size = args.val_batch_size
img_height = args.img_height
img_width = args.img_width
jaccard_weight = args.jaccard_weight  # [0, 1]
model_save = args.out_dir
data_root = args.dataset_dir
pretrained_backbones = args.model_load_dir

def BinaryLoss(outputs, targets, jaccard_weight=1):

    loss = (1- jaccard_weight) * flow.nn.sigmoid_cross_entropy_with_logits(targets,outputs)

    loss = flow.math.reduce_mean(loss)

    eps = 1e-15

    outputs=flow.math.sigmoid(outputs)

    intersection= flow.math.reduce_sum( targets * outputs )

    union = flow.math.reduce_sum(outputs) + flow.math.reduce_sum(targets)

    loss = loss -  jaccard_weight * flow.math.log ( (intersection+eps)/(union-intersection+eps)  )

    return loss

def train(train_data, mask_data, epochs, lr=1e-2):

    @flow.global_function(type="train")
    def train_faceseg_job(image=flow.FixedTensorDef((batch_size,3,img_height,img_width), dtype=flow.float),
                        mask=flow.FixedTensorDef((batch_size,1,img_height,img_width), dtype=flow.float)
    ) :

        feature = LinkNet34(image,trainable=True,batch_size=batch_size) # feed input to model, get output features

        feature= flow.reshape(feature,[-1])
        mask = flow.reshape(mask,[-1])

        # loss function for segmentation
        loss = BinaryLoss(feature,mask,jaccard_weight=jaccard_weight)

        lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(0.01, [100, 200, 300], 0.1)

        flow.optimizer.SGD(lr_scheduler, loss_scale_factor=None,
                           momentum= 0.9, grad_clipping=None, train_step_lbn= None).minimize(loss) # set optimizer
        return loss

    @flow.global_function(type="predict")
    def val_faceseg_job(image=flow.FixedTensorDef((val_batch_size,3,img_height,img_width), dtype=flow.float),
                        mask=flow.FixedTensorDef((val_batch_size,1,img_height,img_width), dtype=flow.float)
    ):

        feature = LinkNet34(image,trainable=False,batch_size=batch_size) # use linknet34 model to segment face

        feature= flow.reshape(feature,[-1])
        mask = flow.reshape(mask,[-1])

        loss = BinaryLoss(feature,mask, jaccard_weight=jaccard_weight) # loss function

        return loss, feature

    loss_l = []
    val_loss_l = []
    best_val_loss = 1e+10
    criterion = Criterion()
    flow.env.log_dir(args.log_dir)


    batch_num = len(train_data) // batch_size # calculate num of iterations
    print('number of Iterations per epoch: ', batch_num)

    # load validation data
    val_data = np.array(np.load(data_root+'img_val.npy'))
    mask_val_data = np.array(np.load(data_root+'mask_val.npy'))

    val_batch_num = len(val_data) // val_batch_size

    # init and load pretrained model
    check_point = flow.train.CheckPoint()
    check_point.load(pretrained_backbones)

    if not os.path.exists(model_save):
        os.mkdir(model_save)

    for epoch in range(epochs):

        epoch_loss  = 0 # reset epoch loss
        val_loss = 0 # reset val loss
        miou = 0
        for batch_idx in range(batch_num):
            inputs = train_data[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ].astype(np.float32, order="C")
            
            target = mask_data[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ].astype(np.float32, order="C")

            # forward get loss
            loss = train_faceseg_job(inputs, target).get()

            epoch_loss += loss.numpy()

        epoch_loss = epoch_loss / (batch_idx + 1)
        loss_l.append(epoch_loss)

        print ("Ep %i train loss : %.3f"%(epoch, epoch_loss))

        for batch_idx in range(val_batch_num):
            inputs = val_data[
                     batch_idx * val_batch_size: (batch_idx + 1) * val_batch_size
                     ].astype(np.float32, order="C")

            target = mask_val_data[
                     batch_idx * val_batch_size: (batch_idx + 1) * val_batch_size
                     ].astype(np.float32, order="C")

            loss, feature = val_faceseg_job(inputs, target).get()

            val_loss += loss.numpy()

            iou_np = criterion.get_miou(feature.numpy(), target)
            miou += iou_np

        miou = miou / (batch_idx + 1)

        val_loss = val_loss / (batch_idx + 1)
        val_loss_l.append(val_loss)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            print('saving ... ' + model_save+ "/faceseg_model"
            check_point.save(model_save + "/faceseg_model")

        print ("Ep %i val loss : %.3f , miou: %.3f \n"%(epoch, val_loss, miou*100))

        fig = plt.figure()
        plt.plot(range(epoch+1), np.array(loss_l))
        plt.savefig(model_save + '/loss.png')

        fig = plt.figure()
        plt.plot(range(epoch+1), np.array(val_loss_l))
        plt.savefig(model_save + '/val_loss.png')


if __name__ == "__main__":
    train_data = np.array(np.load(data_root+'img_train.npy'))
    mask_train_data = np.array(np.load(data_root+'mask_train.npy'))
    train(train_data,mask_train_data,epochs=args.epoch)