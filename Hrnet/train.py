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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #指定第五张卡可见

#flow.config.gpu_device_num(5)

# loss and optimizer
loss = JointsMSELoss()
# loss_metric = tf.metrics.Mean()
pck = PCK()


# accuracy_metric = tf.metrics.Mean()

@flow.global_function(type="train",function_config=func_config)
def train_step(images: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 256, 256, 3), dtype=flow.float32),
               target: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 64, 64, 17), dtype=flow.float32),
               target_weight: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 17, 1), dtype=flow.float32),
               ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy]:

    outputs = HRNet(images, training=True)

    # if isinstance(outputs, list):
    #     losses = loss.call(outputs[0], target, target_weight)
    #     for output in outputs[1:]:
    #         losses += loss.call(output, target, target_weight)
    # else:
    #     output = outputs
    #     losses = loss.call(output, target, target_weight)
    # measure accuracy and record loss
    losses = loss.call(outputs, target, target_weight)

    # Set learning rate as 0.001
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    # Set Adam optimizer
    flow.optimizer.Adam(lr_scheduler, do_bias_correction=False).minimize(losses)

    return losses, outputs, target


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


def train(epoch,dataset_train,dataset_length_train,loss, pck):

    losses = AverageMeter()
    acc = AverageMeter()



    print("****************** train  *****************")
    for i, batch_data in enumerate(dataset_train):
        # measure data loading time
        gt = GroundTruth(cfg, batch_data)
        images, target, target_weight = gt.get_ground_truth()
        images = np.ascontiguousarray(images)
        target = np.ascontiguousarray(target)
        target_weight = np.ascontiguousarray(target_weight)
        # compute output

        loss, outputs, target = train_step(images, target, target_weight)

        # measure accuracy and record loss
        losses.update(loss.item(), images.shape[0])

        _, avg_acc, cnt, pred = pck.call(network_output=outputs, target=target)
        acc.update(avg_acc, cnt)
        # measure elapsed time
        if (i + 1) % cfg.PRINT_FREQ == 0:
            print("{}th epoch, {}/{}th batch, Loss {loss.avg:.10f}, Accuracy {acc.avg:.5f}. ".format(epoch + 1, i + 1,dataset_length_train, loss=losses, acc=acc))
       # if i > 100:
       #     print(i,'break')
       #     break

def validate(epoch,dataset_valid,loss,pck,best_perf):
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
    print("The best acc: {}. ".format(best_perf))
    return losses.avg,acc.avg

def main():

    # Dataset
    print('start load data')
    coco_train = CocoDataset(config_params=cfg, dataset_type="train")
    dataset_train, dataset_length_train = coco_train.generate_dataset()

    coco_valid = CocoDataset(config_params=cfg, dataset_type="valid")
    dataset_valid, _ = coco_valid.generate_dataset()
    print('finish load data')
    
    check_point = flow.train.CheckPoint()
    
    
    if cfg.LOAD_WEIGHTS_BEFORE_TRAINING:
        assert os.path.isdir(cfg.save_weights_dir)
        print('start load model')
        check_point.load(cfg.save_weights_dir)
        print('finished load model')
    best_perf = 0.
    pre_epoch = -1

    begin_epoch = cfg.LOAD_WEIGHTS_FROM_EPOCH

    print("****************** start training *****************")
    for epoch in range(begin_epoch,  cfg.EPOCHS):

        start = time.time()
        train(epoch,dataset_train,dataset_length_train,loss, pck)
        print("Time for epoch {} is {} sec.".format(epoch+ 1, time.time() - start))

        # evaluate on validation set
        loss_val, acc_val = validate(epoch,dataset_valid,loss, pck,best_perf)

        if epoch + 1 > 1 and acc_val > best_perf:
            best_perf =  acc_val
            if pre_epoch != -1:
                # delete the previous best checkpoint
                print("delete the previous best {}th epoch model".format(pre_epoch))
                shutil.rmtree(os.path.join(cfg.save_weights_dir, "{}th_epoch".format(pre_epoch)))

            # save parameters
            print("start save the best model")
            check_point.save(
                os.path.join(cfg.save_weights_dir, "{}th_epoch".format(epoch + 1))
            )
            pre_epoch = epoch + 1
            print("finished save the best epoch model")

    print("*************** Train {} done *****************")


if __name__ == '__main__':
    main()
