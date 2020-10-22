import os
import struct
import six
import numpy as np
import oneflow.core.record.record_pb2 as ofrecord
import cv2
import oneflow as flow
from PIL import Image, ImageOps
import oneflow.typing as tp

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def load_dataset(data_dir, mode, hr_size, lr_size, npy=True):
    """
        image transform: randomly crop, mirror, normalization(0,1), transpose(bs, img_channel, h, w) and shuffle
    """
    images_dir = os.path.join(data_dir, mode)
    hr_imgs, lr_imgs = [], []
    for idx, d in enumerate(os.listdir(images_dir)):
        d = os.path.join(images_dir, d)
        print(d)
        if not is_image_file(d):
            print("The file is not an image in:{}, so we continune next one.".format(d))
            continue
        img = Image.open(d)

        # resize to 128 x 128 x 3, and randomly crop to 88 x 88 x 3
        r1, r2 = np.random.randint(40, size=2)
        hr_img = img.resize((hr_size + 40, hr_size + 40))
        hr_img = hr_img.crop((r1, r2, r1 + hr_size, r2 + hr_size))

        # shape of lr_img is 22 x 22 x 3
        # resize cropped hr_img using Image.BICUBIC
        lr_img = hr_img.resize((lr_size, lr_size), resample=3)

        if np.random.rand() > 0.5:
            # random mirroring
            hr_img = ImageOps.mirror(hr_img)
            lr_img = ImageOps.mirror(lr_img)

        # normalizing the images to [0, 1]
        hr_img = np.array(hr_img) / 255.
        lr_img = np.array(lr_img) / 255.
        hr_img = hr_img.transpose(2, 0, 1)
        lr_img = lr_img.transpose(2, 0, 1)
        assert hr_img.shape == (3, hr_size, hr_size), hr_img.shape
        assert lr_img.shape == (3, lr_size, lr_size), lr_img.shape

        hr_imgs.append(hr_img) # 425 x 3 x 88 x 88
        lr_imgs.append(lr_img)  # 425 x 3 x 22 x 22

     # shuffle    
    seed = 1024
    np.random.seed(seed)
    np.random.shuffle(hr_imgs)
    np.random.seed(seed)
    np.random.shuffle(lr_imgs)

    if npy:
        hr_imgs_save_path = os.path.join(data_dir, "{}_{}hr_imgs.npy".format(mode, hr_size))
        lr_imgs_save_path = os.path.join(data_dir, "{}_{}lr_imgs.npy".format(mode, lr_size))
        if hr_imgs != None:
            with open(hr_imgs_save_path, "wb") as f:
                print(hr_imgs_save_path)
                np.save(f, hr_imgs)
                f.close()

        if lr_imgs != None:
            with open(lr_imgs_save_path, "wb") as f:
                print(lr_imgs_save_path)
                np.save(f, lr_imgs)
                f.close()



def load_image(image_path):
    # to RGB
    image = cv2.imread(image_path)[:,:,::-1]
    H, W = image.shape[:2]
    image = (np.array(image) / 255.).astype(np.float32)
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image)

    return image, H, W, image_path.split(".", 1)[0]+"_result." + image_path.split(".", 1)[1]
        
            
if __name__ == "__main__":
    crop_size = 88
    upscale_factor = 4
    lr_size = crop_size // upscale_factor
    data_dir = "./data"
    modes = ["val", "train"]
    for mode in modes:
        load_dataset(data_dir, mode, crop_size, lr_size)

        
        


