"""
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import oneflow as flow
import oneflow.typing as tp
from tsn_model import restsn
import argparse
import time
import mmcv
import os.path as osp
import os
import tempfile
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from video_dataset import *

def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('--config', default = 'test_configs/TSN/tsn_kinetics400_2d_rgb_r50_seg3_f1s1.py', help='test config file path')
    parser.add_argument('--launcher',
        choices=['none', 'pytorch', 'mpi', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--out', help='output result file', default='default.pkl')
    parser.add_argument('--use_softmax', action='store_true',
                        help='whether to use softmax score')
    # for oneflow
    parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
    parser.add_argument("--use_tensorrt", dest="use_tensorrt", action="store_true", default=False, required=False, help="inference with tensorrt")
    parser.add_argument("--model_load_dir", type=str, default='./output/save_model', required=False, help="model load directory")
    parser.add_argument("--log_dir", type=str, default="./output", required=False, help="log info save directory")
    parser.add_argument("-image_height", "--image_height", type=int, default=224, required=False)
    parser.add_argument("-image_width", "--image_width", type=int, default=224, required=False)

    args = parser.parse_args()
    return args

args = parse_args()
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
flow.config.gpu_device_num(args.gpu_num_per_node)
if args.use_tensorrt:
        func_config.use_tensorrt(True)

@flow.global_function(func_config)
def tsn_eval_job(image:tp.Numpy.Placeholder((250,3,224,224))):
    features = restsn(image, trainable=False)
    return features

class TSNInference(object):
    def __init__(self):
        check_point = flow.train.CheckPoint()
        check_point.load(args.model_load_dir)
    def inference(self, imgs):
        array = np.ascontiguousarray(imgs)
        print(array.shape)
        feature = tsn_eval_job(array).get()
        #print('feature',feature.numpy())
        result = np.argmax(feature.numpy().flatten())
        return result

def multi_test():
    global args
    predicts = []
    labels = []
    count = 0
    # VideoDataset set config
    ann_file = "data/kinetics400/kinetics400_val_list_videos_small.txt"
    img_prefix = "data/kinetics400/videos_val"
    img_norm_cfg = {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}
    anno_open = open(ann_file, 'r')
    anno_len = len(anno_open.readlines())
    anno_open.close()
    oneflow_dataset = VideoDataset(ann_file, img_prefix, img_norm_cfg)

    flow.env.grpc_use_no_signal()
    flow.env.log_dir(args.log_dir)
    obj = TSNInference()

    wrong_count = 0
    for i in range(anno_len):
        img_group, label = oneflow_dataset[i]
        #print(img_group)
        flow_result = obj.inference(img_group)

        if label!=flow_result:
            wrong_count = wrong_count +1
            print(label,flow_result)
        if i % 100 == 0:
            print('data_batch {}'.format(i))
        count = count + 1
    precision = (anno_len - wrong_count)/anno_len
    return precision

def main():
    global args
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    final_precision = multi_test()
    print("precision is: {}".format(final_precision))

if __name__ == '__main__':
    main()
