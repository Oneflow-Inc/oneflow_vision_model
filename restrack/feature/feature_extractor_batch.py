import os,time
import numpy as np
import cv2

import oneflow as flow
from .reid_model import resreid
from .restrack_reid import *
## todo:
# 将bbox_tlwh, ori_img转化为numpy矩阵，批量处理，提取特征

class Extractor():

    def __init__(self, model_name, load_path, gpu_ids='0', use_gpu=True, height=256, width=128, seed=1, cls=0):
        self.model_name = model_name
        self.load_path = load_path
        self.gpu_ids = gpu_ids
        self.use_gpu = use_gpu
        self.height = height
        self.width = width
        self.seed = seed

        winSize = (20, 20)
        blockSize = (10, 10)
        blockStride = (5, 5)
        cellSize = (5, 5)
        nbins = 9

        if cls != 0:
            self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        else:
            assert os.path.isdir(load_path)
            print("Restoring model from {}.".format(load_path))
            check_point = flow.train.CheckPoint()
            check_point.load(load_path)

    def __call__(self, input, batch_size=10, feature_type = 0):
        '''
        :param input: detected images, numpy array
        feature_type = 0 表示提取reid特征，1 表示提取hog特征
        :return: image features extracted from input
        '''
        if feature_type == 1:
            winStride = (20, 20)
            padding = (0, 0)
            if isinstance(input, list):
                if len(input) == 0:
                    return np.array([])
                features = []
                for ind in input:
                    ind_ = cv2.resize(ind, (100,75), interpolation=cv2.INTER_LINEAR)
                    extracted_feature = self.hog.compute(ind_, winStride, padding)
                    extracted_feature = extracted_feature.T
                    features.append(extracted_feature)
            else:
                input_ = cv2.resize(input, (100,75), interpolation=cv2.INTER_LINEAR)
                features = self.hog.compute(input_, winStride, padding)
                features = features.T
            features = np.vstack(features)
            #print("hog size: ", (features.shape))
            return features
        else:
            if len(input) == 0:
                    return np.array([])
            #print(len(input))
            features = []
            for ind in input:
                datest = one_batch_image_preprocess(ind, 256, 128)
                print('hello0',datest.shape)
                outs = reid_eval_job(datest).get()

                print('out',outs.ndarray_list_[0].shape)
                features.append(outs.ndarray_list_[0])
            features = np.vstack(features)
            return features

if __name__ == "__main__":
    print("hello wolrd!\n")
    # etreactor = Extractor(model_name='osnet_x1_0',
    #                    load_path='/home/kcadmin/user/xz/deep-person-reid/checkpoint/model.pth.tar-460',
    #                    gpu_ids='0, 1')
    #
    # feature = etreactor(test_img_numpy)
    # print(feature.shape)
    # print(type(feature))
    # print(feature)