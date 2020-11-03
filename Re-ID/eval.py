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

# -*- coding:utf-8 -*-
"""
person Re-Identification inference examples
includes two models for choice
"""
# Version: 0.0.1
# Author: scorpio.lu(luyi@zhejianglab.com)
# Data: 06/28/2020
import os
import argparse
from datetime import datetime
import numpy as np
import math
import cv2
from scipy.spatial.distance import cdist
import oneflow as flow
from reid_model import HS_reid_train, resreid_train
from data_loader import Market1501
import oneflow.typing as tp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description="flags for person re-identification")
parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("--model", type=str, default='resreid', required=False, help="resreid or HS-reid")
parser.add_argument("--batch_size", type=int, default=600, required=False)
parser.add_argument("--data_dir", type=str, default='./dataset', required=False, help="dataset directory")
parser.add_argument("-image_height", "--image_height", type=int, default=256, required=False)
parser.add_argument("-image_width", "--image_width", type=int, default=128, required=False)
parser.add_argument("--model_load_dir", type=str, default="./model", required=False,
                    help="model load directory; ./model or ./HS_model")
args = parser.parse_args()

# configs
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
flow.config.gpu_device_num(args.gpu_num_per_node)

# model opts
model = {'resreid': resreid_train, 'HS-reid': HS_reid_train}
# input
input_blob = tp.Numpy.Placeholder((args.batch_size, 3, args.image_height, args.image_width), dtype=flow.float)


def resize_image(img, origin_h, origin_w, image_height, image_width):
    """
       image resize op
       Args:
           img : input RGB images.
           origin_h (int): height of img
           origin_w (int): width of img
           image_height (int) : setting height of resize op
           image_width (int): setting width of resize op
    """
    w = image_width
    h = image_height
    resized = np.zeros((3, image_height, image_width), dtype=np.float32)
    part = np.zeros((3, origin_h, image_width), dtype=np.float32)
    w_scale = (float)(origin_w - 1) / (w - 1)
    h_scale = (float)(origin_h - 1) / (h - 1)

    for c in range(w):
        if c == w - 1 or origin_w == 1:
            val = img[:, :, origin_w - 1]
        else:
            sx = c * w_scale
            ix = int(sx)
            dx = sx - ix
            val = (1 - dx) * img[:, :, ix] + dx * img[:, :, ix + 1]
        part[:, :, c] = val
    for r in range(h):
        sy = r * h_scale
        iy = int(sy)
        dy = sy - iy
        val = (1 - dy) * part[:, iy, :]
        resized[:, r, :] = val
        if r == h - 1 or origin_h == 1:
            continue
        resized[:, r, :] = resized[:, r, :] + dy * part[:, iy + 1, :]
    return resized


def batch_image_preprocess(img_paths, img_height, img_weidth):
    """
       image processing
       Args:
           img_paths (list) : input image paths.
           img_height (int): height of img after processing
           img_weidth (int): width of img after processing
    """
    result_list = []
    base = np.ones([args.image_height, args.image_width])
    norm_mean = [base * 0.485, base * 0.456, base * 0.406]  # imagenet mean
    norm_std = [0.229, 0.224, 0.225]  # imagenet std
    for img_path in img_paths:
        # read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # hwc->chw
        img = img.transpose(2, 0, 1).astype(np.float32)
        # to tensor
        img = img / 255  # /255
        # bgr2rgb
        img[[0, 1, 2], :, :] = img[[2, 1, 0], :, :]

        w = img_weidth
        h = img_height
        origin_h = img.shape[1]
        origin_w = img.shape[2]
        # resize
        resize_img = resize_image(img, origin_h, origin_w, h, w)
        # image normalize
        resize_img[0] = (resize_img[0] - norm_mean[0]) / norm_std[0]
        resize_img[1] = (resize_img[1] - norm_mean[1]) / norm_std[1]
        resize_img[2] = (resize_img[2] - norm_mean[2]) / norm_std[2]
        result_list.append(resize_img)
    results = np.asarray(result_list).astype(np.float32)
    return results


def evaluate(qf, q_pids, q_camids, gf, g_pids, g_camids, max_rank=50):
    """
        evaluate the re-identification performance
        return mAP, cmc rank
        Args:
            qf: query features
            q_pids: query pids
            q_camids: query camera ids
            gf: gallery features
            g_pids: gallery pids
            g_camids: gallery camera ids
            max_rank: max value of ranks
    """
    num_g = len(gf)
    num_q = len(qf)
    print('Computing distance matrix  ...')
    dist = cdist(qf, gf).astype(np.float16)
    dist = np.power(dist, 2).astype(np.float16)
    print('Computing CMC and mAP ...')
    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))
    indices = np.argsort(dist, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


@flow.global_function("predict", func_config)
def reid_eval_job(image: input_blob):
    # get batch features
    features = model[args.model](image, trainable=False)
    return features


class ReIDInference(object):
    """
        Re-Identification model inference
    """

    def __init__(self):
        check_point = flow.train.CheckPoint()
        if args.model_load_dir:
            # load model from model path
            assert os.path.isdir(args.model_load_dir)
            print("Restoring model from {}.".format(args.model_load_dir))
            check_point.load(args.model_load_dir)
        else:
            # model init
            print("Init model on demand.")
            check_point.init()

    def inference(self, imgs):
        # get input image features
        query_images = batch_image_preprocess(imgs, args.image_height, args.image_width)
        batch_times = math.ceil(len(imgs) / args.batch_size)
        features = []
        for i in range(batch_times):
            try:
                array = query_images[i * args.batch_size:(i + 1) * args.batch_size]
                feature = reid_eval_job(array).get()
                features.extend(feature.ndarray_list_[0])
            except:
                array = query_images[-args.batch_size:]
                feature = reid_eval_job(array).get()
                features.extend(feature.numpy_list()[0][i * args.batch_size - len(imgs):])

        return features


def main():
    print("=".ljust(66, "="))
    print("Running {}: num_gpu = {}.".format(args.model, args.gpu_num_per_node))
    print("=".ljust(66, "="))
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
    print("-".ljust(66, "-"))
    print("Time stamp: {}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))

    # load Re-ID dataset
    print("Loading data from {}".format(args.data_dir))
    dataset = Market1501(root=args.data_dir)
    # get query images
    query_img, query_id, query_cam_id = zip(*dataset.query)
    # get gallery images
    gallery_img, gallery_id, gallery_cam_id = zip(*dataset.gallery)

    # inference
    obj = ReIDInference()
    print('extracting query features...')
    query_features = obj.inference(query_img)
    print('extracting query features done...')
    print('extracting gallery features...')
    gallery_features = obj.inference(gallery_img)
    # evaluate
    print('extracting gallery features...')
    cmc, mAP = evaluate(query_features, np.array(query_id), np.array(query_cam_id), gallery_features,
                        np.array(gallery_id), np.array(gallery_cam_id))
    print("=".ljust(30, "=") + " Result " + "=".ljust(30, "="))
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in [1, 5, 10]:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print("=".ljust(66, "="))


if __name__ == "__main__":
    main()
