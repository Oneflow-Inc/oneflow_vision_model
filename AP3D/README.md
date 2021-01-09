# of_AP3D

## Introduction
This repo provides Appearance-Preserving 3D Convolution for Video-based Person Re-identification.
## Get started
### Dataset
python /home
Mars:
Baidu Netdisk:https://pan.baidu.com/s/1XKBdY8437O79FnjWvkjusw  code:ymc5

### Train a model
python /home/luis/of_AP3D/train.py --root /home/luis/of_AP3D/ -d mars --arch ap3dres50 --gpu 0,1 --save_dir log-mars-ap3d --model_load_dir /home/luis//resnet_v15_of_best_model_val_top1_77318

### Test a model
python /home/luis/of_AP3D/test-all.py --root /home/luis/of_AP3D/ -d mars --arch ap3dres50 --gpu 0,1 --save_dir log-mars-ap3d --model_load_dir best_model



### Citation


    @inproceedings{gu2020AP3D,
      title={Appearance-Preserving 3D Convolution for Video-based Person Re-identification},
      author={Gu, Xinqian and Chang, Hong and Ma, Bingpeng and Zhang, Hongkai and Chen, Xilin},
      booktitle={ECCV},
      year={2020},
    }
