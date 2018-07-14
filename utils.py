#
# from https://github.com/fastai/fastai/blob/master/fastai/io.py
#


import os
import glob
import shutil
from urllib.request import urlretrieve
from tqdm import tqdm
import tarfile
import gzip
import time

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)


def get_data(url, filename):
    if not os.path.exists(filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        #     urlretrieve(url, filename, reporthook=t.update_to)
        urlretrieve(url, filename)