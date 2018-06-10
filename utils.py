#
# from https://github.com/fastai/fastai/blob/master/fastai/io.py
#


import os
import shutil
from urllib.request import urlretrieve
from tqdm import tqdm
import tarfile
import time

import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)


def get_data(url, filename):
    if not os.path.exists(filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urlretrieve(url, filename, reporthook=t.update_to)


def download_cifar10(data_path):
    def untar_file(file_path, save_path):
        if file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            obj = tarfile.open(file_path)
            obj.extractall(save_path)
            obj.close()
            os.remove(file_path)

    cifar_url = 'http://files.fast.ai/data/cifar10.tgz'

    start_time = time.time()
    get_data(cifar_url, data_path+'/cifar10.tgz')
    download_time = time.time() - start_time

    start_time = time.time()
    untar_file(data_path+'/cifar10.tgz', data_path)
    untar_time = time.time() - start_time

    # Loader expects train and test folders to be outside of cifar10 folder
    shutil.move(data_path+'/cifar10/train', data_path)
    shutil.move(data_path+'/cifar10/test', data_path)
    return download_time, untar_time


#
# https://github.com/wang-chen/KervNets/blob/master/cifar-10.py
#
def make_cifar10_dataset(data_path, batch_size, test_batch_size, distributed=False, num_workers=0):
    transform_train = torchvision.transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(.25,.25,.25),
        transforms.RandomRotation(2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(root=data_path+"/train/", train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers) #pin_memory=True
    """
    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_dataset)
                     if distributed else None)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    """

    test_data = torchvision.datasets.CIFAR10(root=data_path+"/test/", train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers) #pin_memory=True

    return trainloader, testloader



