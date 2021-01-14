from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import time

import os.path as osp
import numpy as np

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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
        self.avg = self.sum / self.count
    def save_checkpoint(is_best, fpath='checkpoint_models'):
        mkdir_if_missing(osp.dirname(fpath))
        check_point = flow.train.CheckPoint() #构造 CheckPoint 对象
        check_point.init()
        check_point.save(fpath)
        if is_best:
            shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))
def match_top_k(predictions, labels, top_k=1):
    max_k_preds = np.argpartition(predictions.numpy(), -top_k)[:, -top_k:]
    match_array = np.logical_or.reduce(max_k_preds == labels.reshape((-1, 1)), axis=1)
    num_matched = match_array.sum()
    return num_matched, match_array.shape[0]


class StopWatch(object):
    def __init__(self):
        pass

    def start(self):
        self.start_time = time.time()
        self.last_split = self.start_time

    def split(self):
        now = time.time()
        duration = now - self.last_split
        self.last_split = now
        return duration

    def stop(self):
        self.stop_time = time.time()

    def duration(self):
        return self.stop_time - self.start_time
class Metric(object):
    def __init__(self, summary=None, save_summary_steps=-1, desc='train', calculate_batches=-1,
                 batch_size=256, top_k=5, prediction_key='predictions', label_key='labels',
                 loss_key=None):
        self.summary = summary
        self.save_summary = isinstance(self.summary, Summary)
        self.save_summary_steps = save_summary_steps
        self.desc = desc
        self.calculate_batches = calculate_batches
        self.top_k = top_k
        self.prediction_key = prediction_key
        self.label_key = label_key
        self.loss_key = loss_key
        if loss_key:
            self.fmt = "{}: epoch {}, iter {}, loss: {:.6f}, top_1: {:.6f}, top_k: {:.6f}, samples/s: {:.3f}"
        else:
            self.fmt = "{}: epoch {}, iter {}, top_1: {:.6f}, top_k: {:.6f}, samples/s: {:.3f}"

        self.timer = StopWatch()
        self.timer.start()
        self._clear()

    def _clear(self):
        self.top_1_num_matched = 0
        self.top_k_num_matched = 0
        self.num_samples = 0.0

    def metric_cb(self, epoch, step):
        def callback(outputs):
            if step == 0: self._clear()
            if self.prediction_key:
                num_matched, num_samples = match_top_k(outputs[self.prediction_key],
                                                       outputs[self.label_key])
                self.top_1_num_matched += num_matched
                num_matched, _ = match_top_k(outputs[self.prediction_key],
                                             outputs[self.label_key], self.top_k)
                self.top_k_num_matched += num_matched
            else:
                num_samples = outputs[self.label_key].shape[0]

            self.num_samples += num_samples

            if (step + 1) % self.calculate_batches == 0:
                throughput = self.num_samples / self.timer.split()
                if self.prediction_key:
                    top_1_accuracy = self.top_1_num_matched / self.num_samples
                    top_k_accuracy = self.top_k_num_matched / self.num_samples
                else:
                    top_1_accuracy = 0.0
                    top_k_accuracy = 0.0

                if self.loss_key:
                    loss = outputs[self.loss_key].mean()
                    print(self.fmt.format(self.desc, epoch, step + 1, loss, top_1_accuracy,
                                          top_k_accuracy, throughput), time.time())
                    if self.save_summary:
                        self.summary.scalar(self.desc+"_" + self.loss_key, loss, epoch, step)
                else:
                    print(self.fmt.format(self.desc, epoch, step + 1, top_1_accuracy,
                                          top_k_accuracy, throughput), time.time())

                self._clear()
                if self.save_summary:
                    self.summary.scalar(self.desc + "_throughput", throughput, epoch, step)
                    if self.prediction_key:
                        self.summary.scalar(self.desc + "_top_1", top_1_accuracy, epoch, step)
                        self.summary.scalar(self.desc + "_top_{}".format(self.top_k),
                                            top_k_accuracy, epoch, step)

            if self.save_summary:
                if (step + 1) % self.save_summary_steps == 0:
                    self.summary.save()

        return callback
