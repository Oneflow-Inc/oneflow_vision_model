#-*- coding:utf-8 -*-
""" 
 @author: scorpio.lu
 @datetime:2020-06-16 9:19
 @software: PyCharm
 @contact: luyi@zhejianglab.com

            ----------
             路有敬亭山
            ----------
 
"""
import torch
import pickle
from collections import OrderedDict
from functools import partial
import os.path as osp
import os
import warnings
def load_checkpoint(fpath):
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint

def load_pretrained_weights(model, weight_path):
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path))
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(weight_path))
        if len(discarded_layers) > 0:
            print('** The following layers are discarded '
                  'due to unmatched keys or layer size: {}'.format(discarded_layers))


def _SaveWeightBlob2File(blob, folder, var):
    #print(blob.shape, blob.dtype , folder, var)
    if not osp.exists(folder):
        os.makedirs(folder)
    filename = osp.join(folder, var)
    f = open(filename, 'wb')
    f.write(blob.tobytes())
    #f.write(blob.tostring())
    f.close()
    #np.save(filename, blob)

def convert(pth_path, of_path='model_pcb'):
    checkpoint = load_checkpoint(pth_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    for k, v in state_dict.items():
        if 'num_batches_tracked' in k:
            print(k)
            continue
        if 'classifier' in k:
            print(k)
            continue
        k = 'base-' + k
        k = k.replace('.', '-')
        k = k.replace('running_mean', 'moving_mean')
        k = k.replace('running_var', 'moving_variance')
        k = k.replace('base-bn1-weight', 'base-bn1-gamma')
        k = k.replace('base-bn1-bias', 'base-bn1-beta')
        k = k.replace('bottleneck-weight', 'bottleneck-gamma')
        k = k.replace('bottleneck-bias', 'bottleneck-beta')

        if k[5:].startswith('layer'):
            k = k.replace('conv1-weight', 'branch2a-weight')
            k = k.replace('bn1-weight', 'branch2a_bn-gamma')
            k = k.replace('bn1-bias', 'branch2a_bn-beta')
            k = k.replace('bn1', 'branch2a_bn')

            k = k.replace('conv2-weight', 'branch2b-weight')
            k = k.replace('bn2-weight', 'branch2b_bn-gamma')
            k = k.replace('bn2-bias', 'branch2b_bn-beta')
            k = k.replace('bn2', 'branch2b_bn')

            k = k.replace('conv3-weight', 'branch2c-weight')
            k = k.replace('bn3-weight', 'branch2c_bn-gamma')
            k = k.replace('bn3-bias', 'branch2c_bn-beta')
            k = k.replace('bn3', 'branch2c_bn')
            k = k.replace('downsample-0-weight', 'downsample-weight')
            k = k.replace('downsample-1-weight', 'downsample_bn-gamma')
            k = k.replace('downsample-1-bias', 'downsample_bn-beta')
            k = k.replace('downsample-1', 'downsample_bn')

        folder = osp.join(of_path, k)
        _SaveWeightBlob2File(v.numpy(), folder, 'out')


convert("/home/oneflow_reid/person_reid/model.pth.tar-100")