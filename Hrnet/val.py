# import tensorflow as tf
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple
import math
import shutil
import numpy as np
import time

from core.function import train
from core.function import validate

from core.loss import JointsMSELoss
from core.make_dataset import CocoDataset
from core.hrnet import HRNet
from core.make_ground_truth import GroundTruth
from core.metric import PCK
# from test import test_during_training

from configuration.base_config import Config
from utils.tools import get_config_params
#config
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

cfg = get_config_params(Config.TRAINING_CONFIG_NAME)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  #指定第五张卡可见

flow.config.gpu_device_num(4)

# loss and optimizer
loss = JointsMSELoss()
# loss_metric = tf.metrics.Mean()
pck = PCK()


# accuracy_metric = tf.metrics.Mean()


@flow.global_function(function_config=func_config)
def val_step(images: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 256, 256, 3), dtype=flow.float32),
             target: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 64, 64, 17), dtype=flow.float32),
             target_weight: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 17, 1), dtype=flow.float32),
             ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy]:
    outputs = HRNet(images, training=False)

    losses = loss.call(outputs, target, target_weight)

    return losses, outputs, target

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0



def validate(epoch,dataset_valid,loss,pck):
    # batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # end = time.time()
    for i, batch_data in enumerate(dataset_valid):
        # compute output
        gt = GroundTruth(cfg, batch_data)
        images, target, target_weight = gt.get_ground_truth()
        images = np.ascontiguousarray(images)
        target = np.ascontiguousarray(target)
        target_weight = np.ascontiguousarray(target_weight)
        # compute output
        loss, outputs, target = val_step(images, target, target_weight)

        # measure accuracy and record loss
        losses.update(loss.item(), images.shape[0])
        _, avg_acc, cnt, pred = pck.call(network_output=outputs, target=target)

        acc.update(avg_acc, cnt)
        # if (i + 1) % cfg.PRINT_FREQ == 0:
        #     print("{}th epoch, Loss {loss.avg:.10f}, Accuracy {acc.avg:.5f}. ".format(epoch + 1, loss=losses, acc=acc))


    print("****************** evalute  *****************")
    print("{}th epoch, Loss {loss.avg:.10f}, Accuracy {acc.avg:.5f}. ".format(epoch + 1, loss=losses, acc=acc))
    return losses.avg,acc.avg

def main():


    coco_valid = CocoDataset(config_params=cfg, dataset_type="valid")
    dataset_valid, _ = coco_valid.generate_dataset()
    print('finish load data')
    
    check_point = flow.train.CheckPoint()
    
    cfg.LOAD_WEIGHTS_BEFORE_TRAINING = True
    save_weights_dir_new =  '../../../../mnt/local/fengyuchao/HRNet/saved_model/weights/8th_epoch'
    if cfg.LOAD_WEIGHTS_BEFORE_TRAINING:
        assert os.path.isdir(save_weights_dir_new)
        print('start load model')
        check_point.load(save_weights_dir_new)
        print('finished load model')

    print("****************** start training *****************")
 
    loss_val, acc_val = validate(1,dataset_valid, loss, pck)

    print("*************** Train {} done *****************")


if __name__ == '__main__':
    main()
