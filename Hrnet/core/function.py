# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple
import math
import numpy as np

import time
import logging
import os

import numpy as np
# import torch

from core.hrnet import HRNet
from core.make_ground_truth import GroundTruth

from configuration.base_config import Config
from utils.tools import get_config_params
# from configuration.base_config import Config
# from utils.tools import get_config_params

# from core.evaluate import accuracy
# from core.inference import get_final_preds
# from utils.transforms import flip_back
# from utils.vis import save_debug_images

cfg = get_config_params(Config.TRAINING_CONFIG_NAME)
#config
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)



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




def train(epoch,dataset_train,loss,pck):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    @flow.global_function(type="train",function_config=func_config)
    def train_step(images: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 256, 256, 3), dtype=flow.float32),
                   target: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 64, 64, 17), dtype=flow.float32),
                   target_weight: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 17, 1), dtype=flow.float32),
                   ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy]:

        outputs = HRNet(images, training=True)

        if isinstance(outputs, list):
            losses = loss.call(outputs[0], target, target_weight)
            for output in outputs[1:]:
                losses += loss.call(output, target, target_weight)
        else:
            output = outputs
            losses = loss.call(output, target, target_weight)

        # Set learning rate as 0.001
        lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
        # Set Adam optimizer
        flow.optimizer.Adam(lr_scheduler, do_bias_correction=False).minimize(losses)

        # measure accuracy and record loss
        # losses.update(loss.item(), images.size(0))

        return losses, outputs, target

    end = time.time()
    for i, batch_data in enumerate(dataset_train):
        # measure data loading time
        data_time.update(time.time() - end)

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
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [{0}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.10f} ({loss.avg:.10f})\t' \
                  'Accuracy {acc.val:.5f} ({acc.avg:.5f})'.format(
                      epoch+1,  batch_time=batch_time,
                      speed=images.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc))
    return losses,acc
        # if i % config.PRINT_FREQ == 0:
        #     msg = 'Epoch: [{0}][{1}/{2}]\t' \
        #           'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
        #           'Speed {speed:.1f} samples/s\t' \
        #           'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
        #           'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
        #           'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
        #               epoch, i, len(train_loader), batch_time=batch_time,
        #               speed=input.size(0)/batch_time.val,
        #               data_time=data_time, loss=losses, acc=acc)
        #     logger.info(msg)
        #
        #     writer = writer_dict['writer']
        #     global_steps = writer_dict['train_global_steps']
        #     writer.add_scalar('train_loss', losses.val, global_steps)
        #     writer.add_scalar('train_acc', acc.val, global_steps)
        #     writer_dict['train_global_steps'] = global_steps + 1
        #
        #     prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
        #     save_debug_images(config, input, meta, target, pred*4, output,
        #                       prefix)


def validate(dataset_valid, dataset_length_valid,loss,pck ):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()


    num_samples = dataset_length_valid
    all_preds = np.zeros(
        (num_samples, 17, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    # with torch.no_grad():


    @flow.global_function(type="train",function_config=func_config)
    def train_step(images: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 256, 256, 3), dtype=flow.float32),
                   target: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 64, 64, 17), dtype=flow.float32),
                   target_weight: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 17, 1), dtype=flow.float32),
                   ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy]:

        outputs = HRNet(images, training=True)

        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        losses = loss.call(output, target, target_weight)


        # measure accuracy and record loss
        # losses.update(loss.item(), images.size(0))

        return losses, outputs, target

    end = time.time()
    for i, batch_data in enumerate(dataset_valid):
        # compute output
        gt = GroundTruth(cfg, batch_data)
        images, target, target_weight = gt.get_ground_truth()
        images = np.ascontiguousarray(images)
        target = np.ascontiguousarray(target)
        target_weight = np.ascontiguousarray(target_weight)
        # compute output
        loss, outputs, target = train_step(images, target, target_weight)

        # if config.TEST.FLIP_TEST:
        #     # this part is ugly, because pytorch has not supported negative index
        #     # input_flipped = model(input[:, :, :, ::-1])
        #     input_flipped = np.flip(input.cpu().numpy(), 3).copy()
        #     # input_flipped = torch.from_numpy(input_flipped).cuda()
        #     outputs_flipped = model(input_flipped)
        #
        #     if isinstance(outputs_flipped, list):
        #         output_flipped = outputs_flipped[-1]
        #     else:
        #         output_flipped = outputs_flipped
        #
        #     output_flipped = flip_back(output_flipped.cpu().numpy(),
        #                                val_dataset.flip_pairs)
        #     # output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
        #
        #     # feature is not aligned, shift flipped heatmap for higher accuracy
        #     if config.TEST.SHIFT_HEATMAP:
        #         output_flipped[:, :, :, 1:] = \
        #             output_flipped.clone()[:, :, :, 0:-1]
        #
        #     output = (output + output_flipped) * 0.5
        num_images = input.shape[0]
        # measure accuracy and record loss
        losses.update(loss.item(), num_images)
        _, avg_acc, cnt, pred = pck.call(network_output=outputs, target=target)

        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return losses,acc

    #     c = meta['center'].numpy()
    #     s = meta['scale'].numpy()
    #     score = meta['score'].numpy()
    #
    #     preds, maxvals = get_final_preds(
    #         config, output.clone().cpu().numpy(), c, s)
    #
    #     all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
    #     all_preds[idx:idx + num_images, :, 2:3] = maxvals
    #     # double check this all_boxes parts
    #     all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
    #     all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
    #     all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
    #     all_boxes[idx:idx + num_images, 5] = score
    #     image_path.extend(meta['image'])
    #
    #     idx += num_images
    #
    #     if i % config.PRINT_FREQ == 0:
    #         msg = 'Test: [{0}/{1}]\t' \
    #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
    #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
    #               'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
    #             i, len(val_loader), batch_time=batch_time,
    #             loss=losses, acc=acc)
    #         logger.info(msg)
    #
    #         prefix = '{}_{}'.format(
    #             os.path.join(output_dir, 'val'), i
    #         )
    #         save_debug_images(config, input, meta, target, pred * 4, output,
    #                           prefix)
    #
    #     name_values, perf_indicator = val_dataset.evaluate(
    #         config, all_preds, output_dir, all_boxes, image_path,
    #         filenames, imgnums
    #     )
    #
    #     model_name = config.MODEL.NAME
    #     if isinstance(name_values, list):
    #         for name_value in name_values:
    #             _print_name_value(name_value, model_name)
    #     else:
    #         _print_name_value(name_values, model_name)
    #
    #     if writer_dict:
    #         writer = writer_dict['writer']
    #         global_steps = writer_dict['valid_global_steps']
    #         writer.add_scalar(
    #             'valid_loss',
    #             losses.avg,
    #             global_steps
    #         )
    #         writer.add_scalar(
    #             'valid_acc',
    #             acc.avg,
    #             global_steps
    #         )
    #         if isinstance(name_values, list):
    #             for name_value in name_values:
    #                 writer.add_scalars(
    #                     'valid',
    #                     dict(name_value),
    #                     global_steps
    #                 )
    #         else:
    #             writer.add_scalars(
    #                 'valid',
    #                 dict(name_values),
    #                 global_steps
    #             )
    #         writer_dict['valid_global_steps'] = global_steps + 1
    #
    # return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


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
