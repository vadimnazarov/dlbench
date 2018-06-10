import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

import time
import sys
import json
from argparse import ArgumentParser
import multiprocessing as mp

from utils import *
from resnet import make_model


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


def train_cnn_full(model_type, trn_loader, tst_loader, device="cuda:0"):
    assert device != "cpu"
    model = make_model(model_type.lower()).to(device)

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


def train_cnn_gpu_only(model_type, trn_loader, tst_loader, device="cuda:0"):
    assert device != "cpu"
    model = make_model(model_type.lower()).to(device)

    optimizer = optim.Adam(model.parameters())

    dataset = []
    for batch, labels in trn_loader:
        dataset.append((batch.to(device), labels.to(device)))
    for batch, labels in trn_loader:
        dataset.append((batch.to(device), labels.to(device)))

    start = time.time()
    for batch_i, (batch, labels) in enumerate(dataset):
        preds = model(batch)
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()
    end = time.time()

    batch_i += 1
    return round((end - start) / batch_i, 3), batch_i


def train_cnn_ram(model_type, trn_loader, tst_loader, device="cuda:0"):
    assert device != "cpu"
    model = make_model(model_type.lower()).to(device)

    optimizer = optim.Adam(model.parameters())

    dataset = []
    for batch in trn_loader:
        dataset.append(batch)
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
    stats = {"cpu": {}}
    parser = ArgumentParser()
    parser.add_argument("-b", default=256, help="batch size", type=int)
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

        model_list = ["ResNet18", "ResNet152"] 

        print("CIFAR10 benchmark (full pipeline)")
        for model_type in model_list:
            for num_workers in range(0, mp.cpu_count()):
                print("[" + model_type + " #workers ", num_workers, "]", sep="")
                for device in range(torch.cuda.device_count()):
                    trn_loader = make_cifar10_dataset(args.d, args.b, distributed=False, num_workers=num_workers)
                    model_time, n_batches = train_cnn_full(model_type, trn_loader, device)

                    key = "cuda:" + str(device)
                    stats[key] = {}
                    stats[key][model_type] = {}
                    stats[key][model_type]["time"] = model_time
                    stats[key][model_type]["batches"] = n_batches
                    stats[key][model_type]["images"] = args.b * n_batches

                    print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(args.b * n_batches) + " images)")
        print()
        print(stats)

        print("CIFAR10 benchmark (GPU speed only)")
        for model_type in model_list:
            print("[" + model_type + "]")
            for device in range(torch.cuda.device_count()):
                trn_loader = make_cifar10_dataset(args.d, args.b, distributed=False, num_workers=0)
                model_time, n_batches = train_cnn_gpu_only(model_type, trn_loader, device)

                key = "cuda:" + str(device)
                stats[key][model_type]["time"] = model_time
                stats[key][model_type]["batches"] = n_batches
                stats[key][model_type]["images"] = args.b * n_batches

                print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(args.b * n_batches) + " images)")
        print()

        print("CIFAR10 benchmark (RAM -> GPU data transfer)")
        for model_type in model_list:
            print("[" + model_type + "]")
            for device in range(torch.cuda.device_count()):
                trn_loader = make_cifar10_dataset(args.d, args.b, distributed=False, num_workers=0)
                model_time, n_batches = train_cnn_ram(model_type, trn_loader, device)

                key = "cuda:" + str(device)
                stats[key][model_type]["time"] = model_time
                stats[key][model_type]["batches"] = n_batches
                stats[key][model_type]["images"] = args.b * n_batches

                print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(args.b * n_batches) + " images)")
        print()

    with open("log.txt", "w") as outf:
        outf.write(json.dumps(stats, sort_keys=True, indent=4, separators=(',', ': ')))

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