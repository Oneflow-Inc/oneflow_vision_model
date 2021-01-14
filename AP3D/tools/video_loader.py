from __future__ import print_function, absolute_import

import os
import functools
import numpy as np
from PIL import Image
import sys

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def image_loader(path):
    return pil_loader(path)


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video
def get_default_video_loader():
    image_loader = pil_loader
    return functools.partial(video_loader, image_loader=image_loader)
class VideoDataset(object):
    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        index =>tracklets for example:train[0:8298]
        """
        img_paths, pid, camid = self.dataset[index]

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        clip = self.loader(img_paths)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        clip=np.stack(clip,axis=0)
        clip=np.transpose(clip,[1, 0, 2, 3])
        return clip, pid, camid
    def __getbatch__(self, index):
       # img_paths, pid, camid = zip(*self.dataset[index])
        img_paths=[]
        pids=[]
        camids=[]
        for i in index:
            img_path, pid, camid=self.dataset[i]
            img_paths.append(img_path)
            pids.append(pid)
            camids.append(camid)
        pids = list(map(int,pids))
        camids = list(map(int,camids))

        imgs=[]
        final_imgs=[]
        if self.temporal_transform is not None:
            for i in range(len(img_paths)):
                img_paths[i] = self.temporal_transform(img_paths[i])
        for img_path in img_paths:
            img=self.loader(img_path)
            imgs.append(img)              
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            for i in range(len(imgs)):
                temp = [self.spatial_transform(img) for img in imgs[i]]
                temp=np.stack(temp,axis=0)
                temp=np.transpose(temp,[1, 0, 2, 3])
                final_imgs.append(temp)
        final_imgs=np.asarray(final_imgs).astype(np.float32)
  
        return final_imgs, np.array(pids), np.array(camids)

class ImageDataset(object):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        transform (callable, optional): A function/transform that  takes in the
            imgs and transforms it.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, pid, camid) where pid is identity of the clip.
        """
        img_path, pid, camid = self.dataset[index]

        img = image_loader(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid    
