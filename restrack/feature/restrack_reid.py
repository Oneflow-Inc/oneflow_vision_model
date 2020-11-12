#-*- coding:utf-8 -*-
""" 
 @author: scorpio.lu
 @datetime:2020-06-24 10:20
 @software: PyCharm
 @contact: luyi@zhejianglab.com

            ----------
             路有敬亭山
            ----------
 
"""
import oneflow as flow
import numpy as np
import cv2
import os
from .reid_model import resreid
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
#flow.config.gpu_device_num(args.gpu_num_per_node)

batch_size =1
# batch_size>=max(len(each_frame(bbox)))
#input_blob = flow.MirroredTensorDef((batch_size, 3, 256, 128), dtype=flow.float)
input_blob = flow.FixedTensorDef((batch_size, 3, 256, 128), dtype=flow.float)

def resize_image(img, origin_h, origin_w, image_height, image_width):
    w = image_width
    h = image_height
    resized=np.zeros((3, image_height, image_width), dtype=np.float32)
    part=np.zeros((3, origin_h, image_width), dtype = np.float32)
    w_scale = (float)(origin_w - 1) / (w - 1)
    h_scale = (float)(origin_h - 1) / (h - 1)

    for c in range(w):
        if c == w-1 or origin_w == 1:
            val = img[:, :, origin_w-1]
        else:
            sx = c * w_scale
            ix = int(sx)
            dx = sx - ix
            val = (1 - dx) * img[:, :, ix] + dx * img[:, :, ix+1]
        part[:, :, c] = val
    for r in range(h):
        sy = r * h_scale
        iy = int(sy)
        dy = sy - iy
        val = (1-dy)*part[:, iy, :]
        resized[:, r, :] = val
        if r==h-1 or origin_h==1:
            continue
        resized[:, r, :] = resized[:, r, :] + dy * part[:, iy+1, :]
    return resized

def batch_image_preprocess(imgs, img_height, img_width):
    result_list = []
    base = np.ones([img_height, img_width])
    norm_mean = [base * 0.485, base * 0.456, base * 0.406]  # imagenet mean
    norm_std = [0.229, 0.224, 0.225]  # imagenet std
    for img in imgs:
        img = img.transpose(2, 0, 1).astype(np.float32)  # hwc->chw
        img = img / 255  # /255  # to tensor
        img[[0, 1, 2], :, :] = img[[2, 1, 0], :, :]  # bgr2rgb

        w = img_width
        h = img_height
        origin_h = img.shape[1]
        origin_w = img.shape[2]

        resize_img = resize_image(img, origin_h, origin_w, h, w)
        # normalize

        resize_img[0] = (resize_img[0] - norm_mean[0])/ norm_std[0]
        resize_img[1] = (resize_img[1] - norm_mean[1]) / norm_std[1]
        resize_img[2] = (resize_img[2] - norm_mean[2]) / norm_std[2]
        result_list.append(resize_img)
    results = np.asarray(result_list).astype(np.float32)

    return results

def one_batch_image_preprocess(img, img_height, img_width):
    result_list = []
    base = np.ones([img_height, img_width])
    norm_mean = [base * 0.485, base * 0.456, base * 0.406]  # imagenet mean
    norm_std = [0.229, 0.224, 0.225]  # imagenet std
 
    img = img.transpose(2, 0, 1).astype(np.float32)  # hwc->chw
    img = img / 255  # /255  # to tensor
    img[[0, 1, 2], :, :] = img[[2, 1, 0], :, :]  # bgr2rgb

    w = img_width
    h = img_height
    origin_h = img.shape[1]
    origin_w = img.shape[2]

    resize_img = resize_image(img, origin_h, origin_w, h, w)
    # normalize

    resize_img[0] = (resize_img[0] - norm_mean[0])/ norm_std[0]
    resize_img[1] = (resize_img[1] - norm_mean[1]) / norm_std[1]
    resize_img[2] = (resize_img[2] - norm_mean[2]) / norm_std[2]
    result_list.append(resize_img)
    results = np.asarray(result_list).astype(np.float32)

    return results

@flow.global_function(func_config)
def reid_eval_job(image=input_blob):
    features = resreid(image, trainable=False)
    return features


def main():
    print("Loading data from {}")
    dataset = np.random.randint(0, 255, 2*64*32*3).reshape((2, 64, 32, 3))
    print(dataset.shape)
    model_load_dir = '../model_restrack'
    assert os.path.isdir(model_load_dir)
    print("Restoring model from {}.".format(model_load_dir))
    check_point = flow.train.CheckPoint()
    check_point.load(model_load_dir)

    print('extracting features...')
    dataset = batch_image_preprocess(dataset, 256, 128)
    feature = reid_eval_job([dataset]).get()
    print(feature.ndarray_list_[0])
    return feature.ndarray_list_[0]

if __name__ == "__main__":
    main()