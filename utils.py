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


def download_cifar10(data_path):
    def untar_file(file_path, save_path):
        if file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            obj = tarfile.open(file_path)
            obj.extractall(save_path)
            obj.close()
            os.remove(file_path)

    cifar_url = 'http://files.fast.ai/data/cifar10.tgz'

    download_time = -1
    untar_time = -1
    if not os.path.exists(data_path+'/train'):
        if not os.path.exists(data_path+'/cifar10.tgz'):
            print("Downloading CIFAR10...")
            start_time = time.time()
            get_data(cifar_url, data_path+'/cifar10.tgz')
            download_time = time.time() - start_time

        print("Extracting CIFAR10...")
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
def make_cifar10_dataset(data_path, batch_size, distributed=False, num_workers=0):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(.25,.25,.25),
        transforms.RandomRotation(2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_folder = os.path.join(data_path, 'train')
    train_data = torchvision.datasets.ImageFolder(train_folder, transform_train)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    """
    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_dataset)
                     if distributed else None)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    """

    # test_folder = os.path.join(data_path, 'test')
    # test_data = torchvision.datasets.ImageFolder(test_folder, transform_test)
    # testloader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return trainloader


# def make_imdb_dataset(data_path, batch_size, num_workers=0):
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.ColorJitter(.25,.25,.25),
#         transforms.RandomRotation(2),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     train_folder = os.path.join(data_path, 'train')
#     train_data = torchvision.datasets.ImageFolder(train_folder, transform_train)
#     trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

#     return trainloader


# def make_cifar10_dataset_dcgan(data_path, batch_size, distributed=False, num_workers=0):
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.ColorJitter(.25,.25,.25),
#         transforms.RandomRotation(2),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     train_folder = os.path.join(data_path, 'train')
#     train_data = torchvision.datasets.ImageFolder(train_folder, transform_train)
#     trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

#     return trainloader



