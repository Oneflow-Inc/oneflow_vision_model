import os
import glob
import numpy as np
from PIL import Image
from imageio import imread


def RandomFlip(sample):
    if (np.random.randint(0, 2) == 1):
        sample['LR'] = np.fliplr(sample['LR']).copy()
        sample['HR'] = np.fliplr(sample['HR']).copy()
        sample['LR_sr'] = np.fliplr(sample['LR_sr']).copy()
    if (np.random.randint(0, 2) == 1):
        sample['Ref'] = np.fliplr(sample['Ref']).copy()
        sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
    if (np.random.randint(0, 2) == 1):
        sample['LR'] = np.flipud(sample['LR']).copy()
        sample['HR'] = np.flipud(sample['HR']).copy()
        sample['LR_sr'] = np.flipud(sample['LR_sr']).copy()
    if (np.random.randint(0, 2) == 1):
        sample['Ref'] = np.flipud(sample['Ref']).copy()
        sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
    return sample


def RandomRotate(sample):
    k1 = np.random.randint(0, 4)
    sample['LR'] = np.rot90(sample['LR'], k1).copy()
    sample['HR'] = np.rot90(sample['HR'], k1).copy()
    sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()
    k2 = np.random.randint(0, 4)
    sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
    sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
    return sample


def Transpose(sample):
    LR, LR_sr, HR, Ref, Ref_sr = sample['LR'], sample['LR_sr'], sample['HR'], sample['Ref'], sample['Ref_sr']
    LR = LR.transpose((2,0,1))
    LR_sr = LR_sr.transpose((2,0,1))
    HR = HR.transpose((2,0,1))
    Ref = Ref.transpose((2,0,1))
    Ref_sr = Ref_sr.transpose((2,0,1))
    return {'LR': LR,
            'LR_sr': LR_sr,
            'HR': HR,
            'Ref': Ref,
            'Ref_sr': Ref_sr}


class TrainSet(object):
    def __init__(self, args):
        self.input_list = sorted([os.path.join(args.data_dir, 'train/input', name) for name in
            os.listdir(os.path.join(args.data_dir, 'train/input'))])
        self.ref_list = sorted([os.path.join(args.data_dir, 'train/ref', name) for name in
            os.listdir(os.path.join(args.data_dir, 'train/ref'))])

    def shuffle(self, seed):
        np.random.seed(seed)
        np.random.shuffle(self.input_list)
        np.random.seed(seed)
        np.random.shuffle(self.ref_list)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = imread(self.input_list[idx])
        h, w = HR.shape[:2]
        # HR = HR[:h//4*4, :w//4*4, :]

        ### LR and LR_sr
        LR = np.array(Image.fromarray(HR).resize((w // 4, h // 4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref_sub = imread(self.ref_list[idx])
        h2, w2 = Ref_sub.shape[:2]
        Ref_sr_sub = np.array(Image.fromarray(Ref_sub).resize((w2 // 4, h2 // 4), Image.BICUBIC))
        Ref_sr_sub = np.array(Image.fromarray(Ref_sr_sub).resize((w2, h2), Image.BICUBIC))

        ### complete ref and ref_sr to the same size, to use batch_size > 1
        Ref = np.zeros((160, 160, 3))
        Ref_sr = np.zeros((160, 160, 3))
        Ref[:h2, :w2, :] = Ref_sub
        Ref_sr[:h2, :w2, :] = Ref_sr_sub

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR': LR,
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref,
                  'Ref_sr': Ref_sr}

        sample = RandomFlip(sample)
        sample = RandomRotate(sample)
        sample = Transpose(sample)
        return sample


class TestSet(object):
    def __init__(self, args):
        self.input_list = sorted(glob.glob(os.path.join(args.data_dir, 'test/CUFED5', '*_0.png')))
        self.ref_list = sorted(glob.glob(os.path.join(args.data_dir, 'test/CUFED5', '*_1.png')))

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = imread(self.input_list[idx])
        HR = np.array(Image.fromarray(HR).resize((160, 160), Image.BICUBIC))

        ### LR and LR_sr
        LR = np.array(Image.fromarray(HR).resize((40, 40), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((160, 160), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref = imread(self.ref_list[idx])
        Ref = Ref[:160, :160, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize((40, 40), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((160, 160), Image.BICUBIC))

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR': LR,
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref,
                  'Ref_sr': Ref_sr}

        sample = Transpose(sample)
        return sample