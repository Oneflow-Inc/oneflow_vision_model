import argparse
import os
from rightmodel import RFRNetModel
from dataset import Dataset
from rightmodel import train


# from torch.utils.data import DataLoader

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',  type=str, default='/mnt/local/fengyuchao/RFR/CelebAMask-HQ/CelebA-HQ-img')
    parser.add_argument('--mask_root', type=str, default='/mnt/local/fengyuchao/RFR/CelebAMask-HQ/CelebAMask-HQ-mask-anno')
    parser.add_argument('--model_save_path', type=str, default='checkpoint')
    parser.add_argument('--result_save_path', type=str, default='results')
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--mask_mode', type=int, default=1)
    parser.add_argument('--num_iters', type=int, default=600000)
    parser.add_argument('--model_path', type=str, default="checkpoint/epoch_0_g_4000")
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--n_threads', type=int, default=6)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu_id', type=str, default="3")
    parser.add_argument('--vgg_path',  type=str, default='/home/qwe/oneflow/bRFR-Inpainting/of_vgg16bn_reuse')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = RFRNetModel()
    if args.test:
        itt=  model.initialize_model(args.model_path, False,args.iter)
        # model.cuda()
        # dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True, training=False))
        dataset = Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse=True,training=False)
        model.test(dataset, args.result_save_path,args.batch_size)
    else:
        itt1 = model.initialize_model(args.model_path, True,args.iter)
        # model.cuda()
        dataset = Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse=True)
        # print(dataset.shape)
        # dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)

        train(dataset, args.model_save_path, args.vgg_path, args.finetune, args.num_iters, args.batch_size,args.epoch,itt1)


if __name__ == '__main__':
    run()