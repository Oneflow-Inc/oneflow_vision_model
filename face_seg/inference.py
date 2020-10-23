
#-*- coding:utf-8 -*-
"""
# Version:0.0.1
# Date:15/10/2020
# Author: Jiaojiao Ye (jiaojiaoye@zhejianglab.com)
"""

import cv2
import numpy as np
from model import faceSeg
import oneflow as flow
import os
import time
from scipy.ndimage import *
from glob import glob
import argparse


# arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Face segmentation')

    # for oneflow
    parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
    parser.add_argument("--model_load_dir", type=str, default='./faceseg_model', required=False, help="model load directory")
    parser.add_argument("--image_dir", type=str, default='./data/example/', required=False, help="demo examples directory")
    parser.add_argument("--img_height", type=int, default=256, required=False)
    parser.add_argument("--img_width", type=int, default=256, required=False)
    parser.add_argument("--jaccard_weight", type=float, default=1 , required=False,  help='jaccard weight for loss, a float between 0 and 1.')

    args = parser.parse_args()
    return args


# test config
args = parse_args()
func_config = flow.function_config()
func_config.default_data_type(flow.float)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
t_l = []
img_height = args.img_height
img_width = args.img_width
jaccard_weight = args.jaccard_weight
model_load_pth = args.model_load_dir

img_dir = args.image_dir
img_pth = glob(img_dir+"*")
name_l = [ img.split('/')[-1]for img in img_pth]
delta_t = []
smooth = True

def plt_mask(name, img_dir):
    # segment face using model
    img_path = img_dir + name  # path of image

    feature = faceSeg(img_path, model_load_pth)

    feature1 = np.squeeze(feature.numpy()) # reshape from (1,1,size,size) to (size,size)

    time_1 = time.time()
    # filter mask contour
    if smooth:
        feature1 = median_filter(feature1, size=5)
    time_2 = time.time()
    print(f'Smooth time: {time_2 - time_1} \n')
    t_l.append(time_2 - time_1)

    face = np.zeros((img_height, img_width, 3), dtype=np.uint8) # face iamge, bgr image

    src = cv2.imread(img_path)  # read source image
    img_test1 = cv2.resize(src, (img_height, img_width))  # resize image
    image = img_test1

    # Mask replace speedup
    feature1 = feature1 >0. # extract mask
    face[feature1==1] = image[feature1==1]

    if not os.path.exists(model_dir + './demo/'):
        os.mkdir(model_dir + 'demo/')
    cv2.imwrite(model_dir + 'demo/' + name, face)



time_1 = time.time()
# load model parameters
check_point = flow.train.CheckPoint()
check_point.load(model_load_pth)
time_2 = time.time()
print(f'Model load time: {time_2 - time_1} \n')

for n in name_l:
    time_1 = time.time()
    plt_mask(n, img_dir)
    time_2 = time.time()
    print(f'time: {time_2 - time_1} \n')
    delta_t.append(time_2 - time_1)

print('Inf time for %i images: %.4fs | Average: %.4fs '%(len(name_l), sum(np.array(t_l)), np.array(t_l).mean()))
print('Execution time for %i images: %.4fs | Average: %.4fs '%(len(name_l), sum(np.array(delta_t)), np.mean(np.array(delta_t))))