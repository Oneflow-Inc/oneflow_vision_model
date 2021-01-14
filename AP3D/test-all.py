from __future__ import print_function, absolute_import
import os
import gc
import sys
import time
import h5py
import scipy
import datetime
import argparse
import os.path as osp
import oneflow as flow
import oneflow.nn as nn
import shutil
import oneflow.typing as tp
from typing import Tuple
import oneflow.math as math
import numpy as np
import models
import models.getresnet as getresnet
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
import tools.data_manager as data_manager
from tools.video_loader import VideoDataset
from tools.losses import TestTripletLoss as TripletLoss
from tools.losses import _CrossEntropyLoss as _CrossEntropyLoss
from tools.utils import AverageMeter, Logger
from tools.eval_metrics import evaluate
from tools.samplers import RandomIdentitySampler
parser = argparse.ArgumentParser(description='Test AP3D using all frames')
# Datasets
parser.add_argument('--root', type=str, default='/content/mars/')
parser.add_argument('-d', '--dataset', type=str, default='mars')
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=128)
# Augment
parser.add_argument("--model_load_dir", type=str, default='/content/resnet_v15_of_best_model_val_top1_77318', required=False,
                    help="model load directory")
parser.add_argument('--seq_len', type=int, default=4, 
                    help="number of images to sample in a tracklet")
parser.add_argument('--sample_stride', type=int, default=8, 
                    help="stride of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max_epoch', default=240, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--train_batch', default=32, type=int)
parser.add_argument('--test_batch', default=32, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--stepsize', default=[14100, 28200, 42300], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float)
parser.add_argument('--margin', type=float, default=0.3, 
                    help="margin for triplet loss")
parser.add_argument('--distance', type=str, default='cosine', 
                    help="euclidean or cosine")
parser.add_argument('--num_instances', type=int, default=4, 
                    help="number of instances per identity")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='ap3dres50', 
                    help="ap3dres50, ap3dnlres50")
# Miscs
parser.add_argument('--eval_step', type=int, default=10)
parser.add_argument('--start_eval', type=int, default=0, 
                    help="start to evaluate after specific epoch")
parser.add_argument('--save_dir', type=str, default='log-mars-ap3d')

parser.add_argument('--gpu', default='0', type=str, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')


args = parser.parse_args()
                    
test_image = tp.Numpy.Placeholder((args.test_batch , 3, args.seq_len, args.height,args.width))
input_pid = tp.Numpy.Placeholder((args.train_batch,))


func_config = flow.FunctionConfig()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
func_config.default_data_type(flow.float)
dataset = data_manager.init_dataset(name=args.dataset, root=args.root)


@flow.global_function(function_config=func_config)
def gallery_job(
    image:test_image
)->tp.Numpy: 
    model = models.init_model(name=args.arch, num_classes=dataset.num_gallery_pids,training=False,resnetblock=getresnet)
    feat=model.build_network(image)
    feat=math.reduce_mean(feat,1)
    feat=flow.layers.batch_normalization(inputs=feat,
                                                axis=1,
                                                 momentum=0.997,
                                                 epsilon=1.001e-5,
                                                 center=True,
                                                 scale=True,
                                                 trainable=False,
                                                 name= "gallery_feature_bn")
    return feat
@flow.global_function(function_config=func_config)
def query_job(
    image:test_image
)->tp.Numpy: 
    model = models.init_model(name=args.arch, num_classes=dataset.num_query_pids,training=False,resnetblock=getresnet)
    feat=model.build_network(image)
    feat=math.reduce_mean(feat,1)
    feat=flow.layers.batch_normalization(inputs=feat,
                                                 momentum=0.997,
                                                 epsilon=1.001e-5,
                                                 center=True,
                                                 scale=True,
                                                 trainable=False,
                                                 axis=1,
                                                 name= "query_feature_bn")
    return feat
    

def getDataSets(dataset):
    spatial_transform_test = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.ToNumpy(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    temporal_transform_test = TT.TemporalBeginCrop()
    queryset = VideoDataset(dataset.query, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test)

    galleryset =VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test)
    return queryset,galleryset

def addmm(mat,mat1,mat2,beta=1,alpha=1):    
    temp=np.matmul(mat,mat2)
    out=(beta*mat+alpha*temp)
    return out




def main():
    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    assert os.path.isdir(args.model_load_dir)
    print("Restoring model from {}.".format(args.model_load_dir))
    checkpoint=flow.train.CheckPoint()
    checkpoint.load(args.model_load_dir)
    queryset,galleryset=getDataSets(dataset)
    print("==> Test")
    rank1=test(queryset,galleryset,dataset)
    




def test(queryset, galleryset, dataset,ranks=[1, 5, 10, 20]):
    since=time.time()
    qf, q_pids, q_camids = [], [], []
    batch_size=args.test_batch
    query_img, query_id, query_cam_id = map(list, zip(*dataset.query))
    indicies=np.arange(len(query_id))
    for i in range(len(indicies) // batch_size):
        try:
            test_batch = queryset.__getbatch__(indicies[i * batch_size:(i + 1) * batch_size])
        except:
            test_batch = queryset.__getbatch__(indicies[-batch_size:])
        feat=query_job(test_batch[0])
        qf.append(feat)
        q_pids.extend(test_batch[1].astype(np.float32))
        q_camids.extend(test_batch[2])
    qf=np.concatenate(qf,0)
    q_pids=np.asarray(q_pids)
    q_camids=np.asarray(q_camids)
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))
    
    gf, g_pids, g_camids = [], [], []
    gallery_img, gallery_id, gallery_cam_id = map(list, zip(*dataset.gallery))
    indicies=np.arange(len(gallery_id))
    for i in range(len(indicies) // batch_size):
        try:
            gallery_batch = galleryset.__getbatch__(indicies[i * batch_size:(i + 1) * batch_size])
        except:
            gallery_batch = galleryset.__getbatch__(indicies[-batch_size:])
        feat=query_job(gallery_batch[0])
        gf.append(feat)
        g_pids.extend(gallery_batch[1].astype(np.float32))
        g_camids.extend(gallery_batch[2])
    gf=np.concatenate(gf,0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    if args.dataset == 'mars':
        gf=np.concatenate((qf,gf),0)
        g_pids = np.append(q_pids, g_pids)
        g_camids = np.append(q_camids, g_camids)

    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print("Computing distance matrix")
    m, n = qf.shape[0], gf.shape[0]
    distmat = np.zeros((m,n))
    if args.distance== 'euclidean':
        distmat1=np.power(qf,2)
        distmat2=np.power(gf,2)
        distmat1=np.sum(distmat1,axis=1,keepdims=True)
        distmat2=np.sum(distmat2,axis=1,keepdims=True)
        distmat1=np.broadcast_to(distmat1,(m,n))
        distmat2=np.broadcast_to(distmat2,(n,m))
        distmat2=np.transpose(distmat2)
        distmat=distmat2+distmat1
        tempgf=np.transpose(gf)

        for i in range(m):
            distmat[i:i+1]=addmm(
                distmat[i:i+1],qf[i:i+1],tempgf,1,-2
            )
    else:
        q_norm=np.linalg.norm(qf,ord=2,axis=1,keepdims=True)
        g_norm=np.linalg.norm(gf,ord=2,axis=1,keepdims=True)
        q_norm=np.broadcast_to(q_norm,qf.shape)
        g_norm=np.broadcast_to(g_norm,gf.shape)
        gf=np.divide(gf,g_norm)
        qf=np.divide(qf,q_norm)
        tempgf=np.transpose(gf)
        for i in range(m):
            distmat[i] = - np.matmul(qf[i:i+1],tempgf)
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0],cmc[4],cmc[9],mAP))
    print("------------------")
    return cmc[0]

if __name__ == '__main__':
    main()








    
    