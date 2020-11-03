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

# Version: 0.0.1
# Author: puchazhong(zhonghw@zhejianglab.com)
# Data: 11/03/2020
import os
import argparse
import numpy as np
import oneflow as flow
from reid_model import resreid_train, HS_reid_train
from data_loader import Market1501, RandomIdentitySampler, ImageDataset
import oneflow.typing as tp
from of_utils.avgmeter import AverageMeter
from of_loss import _TripletLoss, _CrossEntropyLoss
from eval import evaluate
import time

parser = argparse.ArgumentParser(description="flags for person re-identification")
parser.add_argument('--gpu_devices', type=str, default='3')
parser.add_argument("--model", type=str, default='resreid', required=False, help="resreid or HS-reid")
parser.add_argument("--batch_size", type=int, default=64, required=False)
parser.add_argument("--data_dir", type=str, default='./dataset', required=False, help="dataset directory")
parser.add_argument("-image_height", "--image_height", type=int, default=256, required=False)
parser.add_argument("-image_width", "--image_width", type=int, default=128, required=False)
parser.add_argument("--model_load_dir", type=str, default="./pretrained/model_pcb", required=False,
                    help="model load directory")
parser.add_argument("--num_classes", type=int, default=751, required=False)
parser.add_argument("--lr", type=float, default=3.5e-4, required=False)
parser.add_argument("--max_epoch", type=int, default=120, required=False)
parser.add_argument("--step_size", type=list, default=[7360, 12880], required=False)
parser.add_argument("--weight_t", type=float, default=0.5, required=False)
parser.add_argument("--margin", type=float, default=0.3, required=False)
parser.add_argument("--weight_decay", type=float, default=5e-4, required=False)
parser.add_argument("--adam_beta1", type=float, default=0.9, required=False)
parser.add_argument("--adam_beta2", type=float, default=0.999, required=False)
parser.add_argument("--warmup", action='store_true', default=True, help="warm up lr scheduler")
parser.add_argument("--warmup_factor", type=float, default=0.1, required=False)
parser.add_argument("--warmup_iters", type=int, default=1840, required=False)
parser.add_argument("--epsilon", type=float, default=0.1, required=False)
parser.add_argument("--eval_freq", type=int, default=20, required=False)
parser.add_argument("--dist_metric", type=str, default='euclidean', help="euclidean or cosine")
parser.add_argument("--num_instances", type=int, default=4)
parser.add_argument("--eval_batch", type=int, default=600, required=False)
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Modify config options using the command-line')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

# configs
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

# model opts
model = {'resreid': resreid_train, 'HS-reid': HS_reid_train}

# params
max_epoch = args.max_epoch
batch_size = args.batch_size
num_class = args.num_classes
eval_batch = args.eval_batch

# input_size
eval_image = tp.Numpy.Placeholder((args.eval_batch, 3, args.image_height, args.image_width))
input_image = tp.Numpy.Placeholder((args.batch_size, 3, args.image_height, args.image_width))
input_pid = tp.Numpy.Placeholder((args.batch_size,))

# loss
criterion_t = _TripletLoss(margin=args.margin)
criterion_x = _CrossEntropyLoss(num_classes=num_class, epsilon=args.epsilon)
weight_t = args.weight_t
weight_x = 1.0 - args.weight_t


@flow.global_function("train", func_config)
def reid_train_job(image: input_image, pids: input_pid):
    # optimizer init
    warmup_scheduler = flow.optimizer.warmup.linear(args.warmup_iters, args.warmup_factor)
    lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(base_lr=args.lr,
                                                            boundaries=args.step_size,
                                                            scale=[0.1, 0.01],
                                                            warmup=warmup_scheduler)
    opt = flow.optimizer.AdamW(lr_scheduler,
                               beta1=args.adam_beta1,
                               beta2=args.adam_beta2,
                               weight_decay=args.weight_decay)

    features, cls = model[args.model](image, num_class, trainable=True)
    loss_x = criterion_x.forward(cls, pids)
    loss_t = criterion_t.forward(features, pids)
    loss = flow.math.add(flow.math.multiply(weight_t, loss_t), flow.math.multiply(weight_x, loss_x))
    opt.minimize(loss)
    return loss, loss_t, loss_x


@flow.global_function("predict", func_config)
def reid_eval_job(image: eval_image):
    features = model[args.model](image, num_class, trainable=False)
    return features


def inference(dataset):
    # get input image features
    features = []
    ind = list(range(len(dataset)))
    for i in range((len(dataset) // eval_batch) + 1):
        try:
            array, _, _ = dataset.__getbatch__(ind[i * eval_batch:(i + 1) * eval_batch])
            feature = reid_eval_job(array).get()
            features.extend(feature.numpy_list()[0])
        except:
            array, _, _ = dataset.__getbatch__(ind[-eval_batch:])
            feature = reid_eval_job(array).get()
            features.extend(feature.numpy_list()[0][i * eval_batch - len(dataset):])

    return features


def eval(dataset):
    query_img, query_id, query_cam_id = map(list, zip(*dataset.query))
    gallery_img, gallery_id, gallery_cam_id = map(list, zip(*dataset.gallery))
    query_dataset = ImageDataset(dataset.query, flag='test', process_size=(args.image_height, args.image_width))
    gallery_dataset = ImageDataset(dataset.gallery, flag='test', process_size=(args.image_height, args.image_width))
    print("extract query feature")
    time1 = time.time()
    query_features = inference(query_dataset)
    print("extract gallery feature")
    gallery_features = inference(gallery_dataset)
    print("done in {}".format(time.time() - time1))
    return evaluate(query_features, np.array(query_id), np.array(query_cam_id), gallery_features, np.array(gallery_id),
                    np.array(gallery_cam_id))


def train(dataset, batch_size, max_epoch):
    train_img, train_id, train_cam_id = map(list, zip(*dataset.train))
    train_dataset = ImageDataset(dataset.train, flag='train', process_size=(args.image_height, args.image_width))
    for eps in range(max_epoch):
        losses_t = AverageMeter()
        losses_x = AverageMeter()
        losses = AverageMeter()
        indicies = [x for x in RandomIdentitySampler(train_id, batch_size, args.num_instances)]
        for i in range(len(indicies) // batch_size):
            try:
                # train_batch[0,1,2] are [imgs, pid, cam_id]
                train_batch = train_dataset.__getbatch__(indicies[i * batch_size:(i + 1) * batch_size])
            except:
                train_batch = train_dataset.__getbatch__(indicies[-batch_size:])
            loss, loss_t, loss_x = reid_train_job(train_batch[0], train_batch[1].astype(np.float32)).get()

            losses_t.update(loss_t.numpy_list()[0][0], batch_size)
            losses_x.update(loss_x.numpy_list()[0][0], batch_size)
            losses.update(loss.numpy_list()[0][0], batch_size)
        print('epoch: [{0}/{1}]\t'
              'Loss_t {loss_t.val:.4f} ({loss_t.avg:.4f})\t'
              'Loss_x {loss_x.val:.4f} ({loss_x.avg:.4f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            .format(
            eps + 1, args.max_epoch,
            loss_t=losses_t,
            loss_x=losses_x,
            loss=losses))
        if (eps + 1) % args.eval_freq == 0 and (eps + 1) != args.max_epoch:
            cmc, mAP = eval(dataset)
            print("=".ljust(30, "=") + " Result " + "=".ljust(30, "="))
            print('mAP: {:.1%}'.format(mAP))
            print('CMC curve')
            for r in [1, 5, 10]:
                print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
            print("=".ljust(66, "="))
            # print("rank1:{}, mAP:{}".format(cmc[0], mAP))
    print('=> End training')

    print('=> Final test')
    cmc, mAP = eval(dataset)
    print("=".ljust(30, "=") + " Result " + "=".ljust(30, "="))
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in [1, 5, 10]:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print("=".ljust(66, "="))


def main():
    print("=".ljust(66, "="))
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
    print("-".ljust(66, "-"))

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

    # load data for training
    dataset = Market1501(root=args.data_dir)
    train(dataset, batch_size, max_epoch)


if __name__ == "__main__":
    main()
