#-*- coding:utf-8 -*-
""" 
 @author: scorpio.lu
 @datetime:2020-06-11 14:37
 @software: PyCharm
 @contact: luyi@zhejianglab.com

            ----------
             路有敬亭山
            ----------
 
"""
import os
from glob import glob
import re
import errno
import sys
import time
from six.moves import urllib
import tarfile
import zipfile
import numpy as np
import oneflow as flow

class Market1501(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='/home/data', show_summery=True):
        super(Market1501, self).__init__()
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')
        self.download_dataset(self.dataset_dir, self.dataset_url)
        self._check_before_run()

        self.train = self._process_dir(self.train_dir, relabel=True)
        self.query = self._process_dir(self.query_dir, relabel=False)
        self.gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_dataset_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_dataset_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_dataset_info(self.gallery)

        if show_summery:
            print("=> Market1501 has loaded")
            self.print_dataset_statistics()

    def get_dataset_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def _check_before_run(self):
        """Check if all files are available"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def print_dataset_statistics(self):
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(self.num_train_pids, self.num_train_imgs, self.num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(self.num_query_pids, self.num_query_imgs, self.num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams))
        print("  ----------------------------------------")

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append([img_path, pid, camid])

        return dataset

    def download_dataset(self, dir, url):
        """Download and extract the corresponding dataset to the 'dataset_dir' .
        Args:
            dir (str): dataset directory.
            url (str): url where to download dataset.
        """
        if os.path.exists(dir):
            return

        if url is None:
            raise RuntimeError(
                '{} dataset needs to be manually '
                'prepared, please follow the '
                'document to prepare this dataset'.format(
                    self.__class__.__name__
                )
            )

        print('Creating directory "{}"'.format(dir))
        if not os.path.exists(dir):
            try:
                os.makedirs(dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        fpath = os.path.join(dir, os.path.basename(url))
        # download dataset
        print(
            'Downloading {} dataset to "{}"'.format(
                self.__class__.__name__, dir
            )
        )
        self.download_url(url, fpath)
        # extract dataset to the destination
        print('Extracting "{}"'.format(fpath))
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(dir)
            zip_ref.close()

        print('{} dataset is ready'.format(self.__class__.__name__))

    def download_url(self, url, dst):
        """Downloads file from a url to a destination.
        Args:
            url (str): url to download file.
            dst (str): destination path.
        """

        print('* url="{}"'.format(url))
        print('* destination="{}"'.format(dst))

        def _reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(
                '\r...%d%%, %d MB, %d KB/s, %d seconds passed' %
                (percent, progress_size / (1024 * 1024), speed, duration)
            )
            sys.stdout.flush()

        urllib.request.urlretrieve(url, dst, _reporthook)
        sys.stdout.write('\n')