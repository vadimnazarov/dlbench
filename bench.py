import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

import time
import sys
from argparse import ArgumentParser
import multiprocessing as mp

from utils import *
from resnet import *


def train_dnn(batch_size, n_batches, n_features, device="cpu"):
    layers = []
    layers.append(nn.Linear(n_features, 64))
    layers.append(nn.Linear(64, 64))
    layers.append(nn.Linear(64, 64))
    layers.append(nn.Linear(64, 1))
    model = nn.Sequential(*layers)
    model.to(device)

    optimizer = optim.Adam(model.parameters())

    batch_x = torch.FloatTensor([_ for _ in range(n_features * batch_size)]).random_(-1, 1).reshape((batch_size, n_features)).to(device)
    batch_y = torch.normal(torch.FloatTensor([_ for _ in range(batch_size)])).to(device)

    start = time.time()
    for i in range(n_batches):
        pred = model(batch_x)
        loss = F.mse_loss(batch_y, pred)
        loss.backward()
        optimizer.step()
    end = time.time()

    batch_i += 1
    return round((end - start) / (n_batches / 10), 3)


def train_cnn_full(trn_loader, tst_loader, device="cuda:0"):
    assert device != "cpu"
    model = resnet50()
    model.to(device)

    optimizer = optim.Adam(model.parameters())

    start = time.time()
    for batch_i, (batch, labels) in enumerate(trn_loader):
        batch, labels = batch.to(device), labels.to(device)
        preds = model(batch)
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()
    end = time.time()

    batch_i += 1
    return round((end - start) / batch_i, 3), batch_i


def train_cnn_gpu_only(trn_loader, tst_loader, device="cuda:0"):
    assert device != "cpu"
    model = resnet50()
    model.to(device)

    optimizer = optim.Adam(model.parameters())

    dataset = []
    for batch, labels in trn_loader:
        batch, labels = batch.to(device), labels.to(device)
        dataset.append((batch, labels))

    start = time.time()
    for batch_i, (batch, labels) in enumerate(dataset):
        preds = model(batch)
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()
    end = time.time()

    batch_i += 1
    return round((end - start) / batch_i, 3), batch_i


def train_cnn_ram(trn_loader, tst_loader, device="cuda:0"):
    assert device != "cpu"
    model = resnet50()
    model.to(device)

    optimizer = optim.Adam(model.parameters())

    dataset = []
    for batch in trn_loader:
        dataset.append(batch)

    start = time.time()
    for batch_i, (batch, labels) in enumerate(dataset):
        batch, labels = batch.to(device), labels.to(device)
        preds = model(batch)
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()
    end = time.time()

    batch_i += 1
    return round((end - start) / batch_i, 3), batch_i


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", default=64, help="batch size", type=int)
    parser.add_argument("-n", default=100, help="number of batches", type=int)
    parser.add_argument("-f", default=5000, help="number of features", type=int)
    parser.add_argument("-d", default="./data/", help="data folder path", type=str)
    args = parser.parse_args()

    print("Deep Learning Benchmark")
    print("  CUDA:  ", torch.cuda.is_available())
    print("  CUDNN: ", torch.backends.cudnn.enabled)
    print("  #GPUs: ", torch.cuda.device_count())
    print()

    print("CIFAR10 dataset")
    download_time, untar_time = download_cifar10(args.d)
    print("  download time:", round(download_time, 3))
    print("  untar time:", round(untar_time, 3))
    print()

    # print("Simple DNN benchmark")
    # model_time = train_dnn(args.b, args.n, args.f)
    # print("  cpu:", model_time, "sec / 10*batch")

    if torch.cuda.device_count() and torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
        # for device in range(torch.cuda.device_count()):
        #     model_time = train_dnn(args.b, args.n, args.f, device)
        #     print("  cuda:" + str(device), model_time, "sec / 10*batch")
        # print()

        print("CIFAR10 benchmark (full)")
        for num_workers in range(0, mp.cpu_count()):
            print("[ResNet50, #workers ", num_workers, "]", sep="")
            trn_loader = make_cifar10_dataset(args.d, args.b, distributed=False, num_workers=num_workers)
            for device in range(torch.cuda.device_count()):
                model_time, n_batches = train_cnn_full(trn_loader, device)
                print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(args.b * n_batches) + " images)")
        print()

        print("CIFAR10 benchmark (GPU only)")
        print("[ResNet50]")
        trn_loader = make_cifar10_dataset(args.d, args.b, distributed=False, num_workers=0)
        for device in range(torch.cuda.device_count()):
            model_time, n_batches = train_cnn_gpu_only(trn_loader, device)
            print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(args.b * n_batches) + " images)")
        print()

        print("CIFAR10 benchmark (RAM)")
        print("[ResNet50]")
        trn_loader = make_cifar10_dataset(args.d, args.b, distributed=False, num_workers=0)
        for device in range(torch.cuda.device_count()):
            model_time, n_batches = train_cnn_ram(trn_loader, device)
            print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(args.b * n_batches) + " images)")
        print()


"""
0.04
sec / batch

0.04 / 64
sec / (batch * 64) = sec / image

1 / (0.04 / 64)
images / sec

(60 * 60) * 1 / (0.04 / 64)
images * 60 * 60 / sec = images / hour

(60 * 60) * 1 / (0.04 / 64) / OUR_COST
images / $

(60 * 60) * 1 / (0.04 / 64) / 1000000 / OUR_COST
million * images / $

1 / 0.04
batches / sec

60*60 / 0.04
batches / hour

60*60 / OUR_COST
batches / $
"""