import time
import sys
import json
from argparse import ArgumentParser
import multiprocessing as mp

from pandas.io.json import json_normalize

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

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


def train_cnn_full(model_type, trn_loader, device="cuda:0"):
    assert device != "cpu"
    if type(device) is int:
        device = "cuda:" + str(device)
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


def train_cnn_gpu_only(model_type, trn_loader, device="cuda:0"):
    assert device != "cpu"
    if type(device) is int:
        device = "cuda:" + str(device)
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


def train_cnn_ram(model_type, trn_loader, device="cuda:0"):
    assert device != "cpu"
    if type(device) is int:
        device = "cuda:" + str(device)
    model = make_model(model_type.lower()).to(device)

    optimizer = optim.Adam(model.parameters())

    dataset = []
    for batch, labels in trn_loader:
        dataset.append((batch.cpu(), labels.cpu()))
    for batch, labels in trn_loader:
        dataset.append((batch.cpu(), labels.cpu()))

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


def add_item(stats, bench, cuda, model, time, batches, images):
    stats.append({})
    stats[-1]["benchmark"] = bench
    stats[-1]["device"] = cuda
    stats[-1]["model"] = model
    stats[-1]["time"] = time
    stats[-1]["batches"] = batches
    stats[-1]["images"] = images


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", default=256, help="batch size", type=int)
    parser.add_argument("-n", default=100, help="number of batches", type=int)
    parser.add_argument("-f", default=5000, help="number of features", type=int)
    parser.add_argument("-d", default="./data", help="data folder path", type=str)
    parser.add_argument("-o", default="./", help="output folder path", type=str)
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

    stats = []
    if torch.cuda.device_count() and torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True

        model_list = ["ResNet18", "ResNet152"] 

        print("CIFAR10 benchmark (GPU speed only)")
        for model_type in model_list:
            print("[" + model_type + "]")
            for device in range(torch.cuda.device_count()):
                trn_loader = make_cifar10_dataset(args.d, args.b, distributed=False, num_workers=0)
                model_time, n_batches = train_cnn_gpu_only(model_type, trn_loader, device)

                add_item(stats, "speed", "cuda:" + str(device), 
                         model_type, model_time, n_batches, args.b * n_batches)

                print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(args.b * n_batches) + " images)")
        print()

        print("CIFAR10 benchmark (RAM -> GPU data transfer)")
        for model_type in model_list:
            print("[" + model_type + "]")
            for device in range(torch.cuda.device_count()):
                trn_loader = make_cifar10_dataset(args.d, args.b, distributed=False, num_workers=0)
                model_time, n_batches = train_cnn_ram(model_type, trn_loader, device)

                add_item(stats, "transfer", "cuda:" + str(device), 
                         model_type, model_time, n_batches, args.b * n_batches)

                print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(args.b * n_batches) + " images)")
        print()

        print("CIFAR10 benchmark (full pipeline)")
        for model_type in model_list:
            for num_workers in range(0, mp.cpu_count()):
                print("[" + model_type + " #workers ", num_workers, "]", sep="")
                for device in range(torch.cuda.device_count()):
                    trn_loader = make_cifar10_dataset(args.d, args.b, distributed=False, num_workers=num_workers)
                    model_time, n_batches = train_cnn_full(model_type, trn_loader, device)

                    add_item(stats, "full" + str(num_workers), "cuda:" + str(device), 
                             model_type, model_time, n_batches, args.b * n_batches)

                    print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(args.b * n_batches) + " images)")
        print()

    df = json_normalize(stats)
    df.sort_values(by="device", inplace=True)
    print(df)
    df.to_csv(args.o + "/logs.txt")

"""
def comp():
    sec_per_batch = float(input("sec per batch: "))
    cost_per_hour = float(input("cost per hour: "))
    print("batches per dollar:", round(3600 / (sec_per_batch * cost_per_hour), 3))


"""