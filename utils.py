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
def make_cifar10_dataset(data_path, batch_size, device, distributed=False, num_workers=0):
    torch.cuda.device(device)

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



class SentimentRNN(nn.Module):

    def __init__(self, alphabet_size):
        super(SentimentRNN, self).__init__()
        hidden = 32
        n_layers = 1
        self.rnn = nn.GRU(alphabet_size, hidden, n_layers, batch_first=True)
        self.final = nn.Sequential(nn.BatchNorm1d(hidden), 
                                   nn.RReLU(), 
                                   nn.Linear(hidden, 1), 
                                   nn.Sigmoid())

    def forward(self, input):
        _, x = self.rnn(input)
        return self.final(x.squeeze(0)).squeeze(1)


def make_imdb_dataset(data_path):
    print("IMDB: extracting files...")
    if not os.path.exists(data_path + "/imdb"):
        obj = tarfile.open(data_path + "/imdb.tgz")
        obj.extractall(data_path)
        obj.close()
    
    print("IMDB: preprocessing sequences...")
    alphabet_d = {}
    raw_seq_data = []
    lens_data = [] 
    label_data = []
    for key in ["pos", "neg"]:
        for filename in glob.glob("data/imdb/" + key + "/*"):
            with open(filename) as f:
                content = f.read().strip().lower()
                if len(content) < 3000:
                    raw_seq_data.append(list(content))
                    for key in raw_seq_data[-1]:
                       alphabet_d[key] = alphabet_d.get(key, 0) + 1
                    lens_data.append(len(raw_seq_data[-1]))
                    label_data.append(1 if key == "pos" else 0)

    # remove low-frequency letters
    alphabet = list(filter(lambda x: alphabet_d[x] > 2000, alphabet_d))
    other_symbol = "@"
    pad_symbol = "$"
    alphabet.append(pad_symbol)
    alphabet.append(other_symbol)
    char_indices = {x:i for i,x in enumerate(sorted(alphabet))}
    get_char_index = lambda x: char_indices.get(x, char_indices[other_symbol])

    max_len = max(lens_data)

    print("IMDB: converting sequences to tensors...")
    # seq_data = torch.zeros((len(raw_seq_data), max_len, len(alphabet)), dtype=torch.float)
    seq_data = torch.zeros((500, max_len, len(alphabet)), dtype=torch.float)
    for seq_i in range(seq_data.shape[0]):
        for symb_i in range(lens_data[seq_i]):
            seq_data[seq_i, symb_i, get_char_index(seq_data[seq_i][symb_i])] = 1

    lens_data = torch.IntTensor(lens_data[:500])
    label_data = torch.FloatTensor(label_data[:500])

    train_data = torch.utils.data.TensorDataset(seq_data, lens_data, label_data)

    return train_data, len(alphabet)


def make_imdb_dataloader(train_data, batch_size, device, num_workers=0):
    torch.cuda.device(device)
    return torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)


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



