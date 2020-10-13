import oneflow as flow
import oneflow.typing as tp
from tsn_model import restsn
import argparse
import time
import mmcv
import os.path as osp
import os
import sys
import tempfile
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from video_dataset import *
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('--launcher',
        choices=['none', 'pytorch', 'mpi', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--out', help='output result file', default='default.pkl')
    parser.add_argument('--num_classes', help='number of class', type=int, default=400)
    parser.add_argument('--use_softmax', action='store_true',
                        help='whether to use softmax score')
    parser.add_argument("--train_num_segments", type=int, default=3, required=False)    
    parser.add_argument("--epoch", type=int, default=100, required=False)                 

    # for oneflow
    parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
    parser.add_argument("--lr", type=int, default=0.1, required=False)
    parser.add_argument("--use_tensorrt", dest="use_tensorrt", action="store_true", default=False, required=False, help="inference with tensorrt")
    parser.add_argument("--model_load_dir", type=str, default='/home/zjlab/liuxy/mmaction/modelzoo/model_pcb', required=False, help="model load directory")
    parser.add_argument("--log_dir", type=str, default="./output", required=False, help="log info save directory")
    parser.add_argument("--image_height", type=int, default=224, required=False)
    parser.add_argument("--image_width", type=int, default=224, required=False)
    parser.add_argument("--train_batch_size", type=int, default=8, required=False)
    parser.add_argument("--out_dir", type=str, default="output/save_model/", required=False)

    args = parser.parse_args()
    return args

# train config
args = parse_args()
func_config = flow.function_config()
func_config.default_data_type(flow.float)
flow.config.gpu_device_num(args.gpu_num_per_node)

if args.use_tensorrt:
        func_config.use_tensorrt(True)

@flow.global_function('train',func_config)
def train_tsn(image:tp.Numpy.Placeholder((args.train_batch_size*args.train_num_segments,3,224,224)), 
              label:tp.Numpy.Placeholder((args.train_batch_size,400))):
    features = restsn(image, args.train_batch_size, trainable=True)

    loss = flow.nn.softmax_cross_entropy_with_logits(label, features, name="loss_liu")
    # loss = flow.nn.sparse_softmax_cross_entropy_with_logits(label, features, name="loss_liu")
    # set learning rate as 0.1
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.00001])
    # Set SGD optimizer, weight_decay factor is 0.00001
    gradient_clip = flow.optimizer.grad_clipping.by_global_norm(40.0)
    flow.optimizer.AdamW(lr_scheduler,
            weight_decay=0.0001,grad_clipping=gradient_clip).minimize(loss)
    # flow.optimizer.Adam(lr_scheduler,
    #     grad_clipping=gradient_clip).minimize(loss)
    return loss

@flow.global_function('predict')
def val_tsn(image:tp.Numpy.Placeholder((250,3,224,224))):
    output= restsn(image, 1, trainable=False)
    # loss = flow.nn.softmax_cross_entropy_with_logits(label, output, name="loss_liu")
    return output
check_point = flow.train.CheckPoint()
check_point.load(args.model_load_dir)
class TSNTrain(object):
    def __init__(self):
        train_mode = True
    def train(self, imgs, labels):
        img_group = np.ascontiguousarray(imgs)
        bs = img_group.shape[0]
        img_group = img_group.reshape(
            (-1, 3) + img_group.shape[3:])
        num_seg = img_group.shape[0] // bs
        one_hot_labels = (np.arange(args.num_classes)==labels[:,None]).astype(np.float32)
        loss = train_tsn(img_group, one_hot_labels).get()

        return loss.numpy()

def multi_train():
    global args
    count = 0
    # train VideoDataset config
    ann_file = "data/kinetics400/kinetics400_train_list_videos.txt"
    img_prefix = "/data/liuxy/videos_train_big"
    img_norm_cfg = {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}
    anno_open = open(ann_file, 'r')
    anno_len = len(anno_open.readlines())
    anno_open.close()
    oneflow_dataset = VideoDataset(ann_file, 
        img_prefix, 
        img_norm_cfg, 
        num_segments=args.train_num_segments, 
        new_length=1,
        new_step=1,
        random_shift=True,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        flip_ratio=0.5,
        resize_keep_ratio=True,
        oversample=None,
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=True,
        scales=[1, 0.875, 0.75, 0.66],
        max_distort=1,
        test_mode=False)

    # val VideoDataset config
    ann_file_val = "data/kinetics400/kinetics400_val_list_videos.txt"
    img_prefix_val = "/data/liuxy/videos_val_mp4"
    anno_open_val = open(ann_file_val, 'r')
    anno_len_val = len(anno_open_val.readlines())
    anno_open_val.close()
    oneflow_dataset_val = VideoDataset(ann_file_val, img_prefix_val, img_norm_cfg)

    flow.env.grpc_use_no_signal()
    flow.env.log_dir(args.log_dir)

    obj = TSNTrain()
    val_prec = 0
    
    for j in range(args.epoch):
        wrong_count = 0
        iter_num = anno_len//args.train_batch_size

        for i in range(iter_num):
            img_group = []
            labels = []
            for k in range(args.train_batch_size):
                try:
                    img, label = oneflow_dataset[args.train_batch_size*i+k]
                    img_group.append(img)
                    labels.append(label)
                except:
                    print(args.train_batch_size*i+k)
                    break
            img_group = np.array(img_group)
            labels = np.array(labels)
            loss = obj.train(img_group, labels)

            if i % 20 == 0:
                print("Epoch: {}\t iter: [{}/{}]\t loss: {}\t".format(j, i, iter_num, loss.mean()))
        # val after each epoch
        for i in range(anno_len_val):
            img_group, label = oneflow_dataset_val[i]
            array = np.ascontiguousarray(img_group)
            feature = val_tsn(array).get()
            flow_result = np.argmax(feature.numpy().flatten())
            if label!=flow_result:
                wrong_count = wrong_count +1
            count = count + 1
        final_precision = float(anno_len_val - wrong_count)/anno_len_val
        print("val precision is: {}".format(final_precision))
        
        if final_precision > val_prec:
            val_prec = final_precision
            shutil.rmtree(args.out_dir)
            os.mkdir(args.out_dir)
            check_point.save(args.out_dir)
            print("saving model...")

def main():
    global args
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    multi_train()

if __name__ == '__main__':
    main()
