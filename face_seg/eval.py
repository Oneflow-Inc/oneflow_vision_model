"""
# Version:0.0.1
# Date: 15/10/2020
# Author: Jiaojiao Ye (jiaojiaoye@zhejianglab.com)
"""

import oneflow as flow
import numpy as np
import matplotlib.pyplot as plt
import os

# customize function
from train import BinaryLoss
from model import LinkNet34
import argparse

# arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Face segmentation')

    # for oneflow
    parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
    parser.add_argument("--model_load_dir", type=str, default='./output/save_model/faceseg_model', required=False, help="model load directory")
    parser.add_argument("--dataset_dir", type=str, default='./data/', required=False, help="dataset root directory")
    parser.add_argument("--img_height", type=int, default=256, required=False)
    parser.add_argument("--img_width", type=int, default=256, required=False)
    parser.add_argument("--train_batch_size", type=int, default=128, required=False)
    parser.add_argument("--val_batch_size", type=int, default=128, required=False)
    parser.add_argument("--jaccard_weight", type=float, default=1 , required=False,  help='jaccard weight for loss, a float between 0 and 1.')

    args = parser.parse_args()
    return args

# test config
args = parse_args()
func_config = flow.function_config()
func_config.default_data_type(flow.float)
flow.config.gpu_device_num(args.gpu_num_per_node)

os.environ['CUDA_VISIBLE_DEVICES'] = ' 2'
data_dir = args.dataset_dir
img_height = args.img_height
img_width = args.img_width
batch_size = args.train_batch_size
val_batch_size = args.val_batch_size
jaccard_weight = args.jaccard_weight
model_pth = args.model_load_dir


@flow.global_function(type="predict")
def val_faceseg_job(image=flow.FixedTensorDef((val_batch_size, 3, img_height, img_width), dtype=flow.float),
                    mask=flow.FixedTensorDef((val_batch_size, 1, img_height, img_width), dtype=flow.float)
                    ):

    feature = LinkNet34(image, trainable=False, batch_size=val_batch_size)  # use linknet34 model to segment face

    loss = BinaryLoss(feature, mask, jaccard_weight=jaccard_weight)

    return loss, feature

# evaluation the mIoU
class Criterion():

    def __init__(self, jaccard_weight=0):
        self.jaccard_weight = jaccard_weight
        self.num_classes = 2
        self.hist = np.zeros((self.num_classes, self.num_classes))

    '''
    Implementation by: https://github.com/LeeJunHyun/Image_Segmentation
    '''
    def get_miou(self, pred, target):
        # pred: output of network, shape of (batch_size, 1, img_size, img_size)
        # target: true mask, shape of (batch_size, 1, img_size, img_size)

        pred = np.reshape(pred, (batch_size,-1))
        pred  = pred > 0. # get the predict label, positive as label
        target= np.reshape(target, (batch_size, -1))
        inter = np.logical_and(pred, target,)
        union = np.logical_or(pred, target)
        iou_np = np.sum(inter, axis=-1) / (np.sum(union, axis=-1) + 1e-6) # iou equation, add 1e-6 to avoid zero division
        iou_np = np.mean(iou_np)
        return iou_np


def evaluate():
    # evaluate iou and loss of the model

    # load train and validate data
    train_data = np.array(np.load(data_dir+'img_train.npy'))
    mask_train_data = np.array(np.load(data_dir+'mask_train.npy'))
    val_data = np.array(np.load(data_dir+'img_val.npy'))
    mask_val_data = np.array(np.load(data_dir+'mask_val.npy'))

    # load model
    check_point = flow.train.CheckPoint()
    check_point.load(model_pth)
    # check_point.init()

    # Eval on train data
    train_batch_num = len(train_data) // batch_size
    train_loss = 0
    miou = 0

    criterion = Criterion()

    for batch_idx in range(train_batch_num):
        inputs = train_data[
                 batch_idx * batch_size: (batch_idx + 1) * batch_size
                 ].astype(np.float32, order="C")

        target = mask_train_data[
                 batch_idx * batch_size: (batch_idx + 1) * batch_size
                 ].astype(np.float32, order="C")

        loss,feature = val_faceseg_job(inputs, target).get()

        train_loss += loss.numpy()

        iou_np = criterion.get_miou(feature.numpy(), target)
        miou += iou_np

    miou = miou /(batch_idx+1)

    train_loss = train_loss / (batch_idx + 1)
    print ("Train loss of model %s : %.3f"%( model_pth, train_loss))
    print ("Train MIoU of model %s : %.3f "%( model_pth, miou *100))

    # Evaluate on validation data
    val_batch_num = len(val_data) // val_batch_size
    val_loss = 0
    miou = 0

    for batch_idx in range(val_batch_num):
        inputs = val_data[
                 batch_idx * val_batch_size: (batch_idx + 1) * val_batch_size
                 ].astype(np.float32, order="C")

        target = mask_val_data[
                 batch_idx * val_batch_size: (batch_idx + 1) * val_batch_size
                 ].astype(np.float32, order="C")

        loss,feature = val_faceseg_job(inputs, target).get()

        val_loss += loss.numpy()

        iou_np = criterion.get_miou(feature.numpy(), target)

        miou += iou_np
    miou = miou /(batch_idx+1)

    val_loss = val_loss / (batch_idx + 1)
    print ("Val loss of model %s : %.3f"%( model_pth, val_loss))
    print ("Val MIoU of model %s : %.3f "%( model_pth, miou *100))
    print('')

if __name__ == '__main__':
    evaluate()