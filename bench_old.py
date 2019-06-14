import time
import sys
import json
from argparse import ArgumentParser
import multiprocessing as mp

import numpy as np
from pandas.io.json import json_normalize

import torch
from cnn import *
from sentim import *
from wgan import *
# import resnet
# import sentim


def add_item(stats, bench, cuda, model, time, batches, images):
    stats.append({})
    stats[-1]["benchmark"] = bench
    stats[-1]["device"] = cuda
    stats[-1]["model"] = model
    stats[-1]["time"] = time
    stats[-1]["batches"] = batches
    stats[-1]["objects"] = images


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", default=256, help="batch size for ResNet CIFAR10", type=int)
    parser.add_argument("--br", default=128, help="batch size for RNN", type=int)
    parser.add_argument("--bg", default=64, help="batch size for WGAN", type=int)
    parser.add_argument("-n", default=100, help="number of batches", type=int)
    parser.add_argument("-f", default=5000, help="number of features", type=int)
    parser.add_argument("-d", default="./data", help="data folder path", type=str)
    parser.add_argument("-o", default="./", help="output folder path", type=str)
    parser.add_argument("-c", "--cuda", default="all", help="which cuda devices use (example - '0,1,3', default - 'all')", type=str)
    parser.add_argument("--mc", "--max_cpu", default=8, help="max number of CPU cores to use", type=int)
    args = parser.parse_args()

    if args.cuda == "all":
        cuda_devices = list(range(torch.cuda.device_count()))
    else:
        cuda_devices = list(map(lambda x: int(x.strip()), args.cuda.split(",")))

    print("Deep Learning Benchmark")
    print("  CUDA:  ", torch.cuda.is_available())
    print("  CUDNN: ", torch.backends.cudnn.enabled)
    print("  #GPUs: ", torch.cuda.device_count())
    print("  GPUs selected: ", cuda_devices)
    print()

    print("CIFAR10 dataset")
    download_time, untar_time = download_cifar10(args.d)
    print("  download time:", round(download_time, 3))
    print("  untar time:", round(untar_time, 3))
    print()

#     sentiment_data, alphabet_size = make_imdb_dataset(args.d)
    print()

    stats = []
    # max_cpu_count = min(mp.cpu_count(), 8) + 1
    max_cpu_count = min(mp.cpu_count(), args.mc) + 1
    if torch.cuda.device_count() and torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True

        # print("Neural style benchmark (GPU mostly)")
        # for device in cuda_devices:
        #     model_time, n_batches = train_neural_style(device)

        #     add_item(stats, "style", "cuda:" + str(device), 
        #              "VGG", model_time, n_batches, 2 * n_batches)

        #     print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(2 * n_batches) + " images)")
        # print()

        ################################################
        # WGAN training
        ################################################
        # print("WGAN benchmark (full)")
        # for num_workers in range(0, 2):
        #     print("[WGAN #workers ", num_workers, "]", sep="")

        #     for device in cuda_devices:
        #         trn_loader = make_cifar10_dataset_wgan(args.d, args.bg, device=device, num_workers=num_workers)
        #         model_time, n_batches = train_wgan(trn_loader, device)

        #         add_item(stats, "wgan" + str(num_workers).zfill(2), "cuda:" + str(device), 
        #                  "WGAN", model_time, n_batches, args.bg * n_batches)

        #         print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(args.bg * n_batches) + " images)")
        # print()
    

        ################################################
        # Sentiment analysis
        ################################################
        # print("Sentiment analysis benchmark (full)")
        # for num_workers in range(0, min(5, max_cpu_count)):
        #     print("[GRU #workers ", num_workers, "]", sep="")

        #     for device in cuda_devices:
        #         trn_loader = make_imdb_dataloader(sentiment_data, args.br, device, num_workers=num_workers)
        #         model_time, n_batches = train_sentiment(trn_loader, alphabet_size, device)

        #         add_item(stats, "sentim" + str(num_workers).zfill(2), "cuda:" + str(device), 
        #                  "GRU", model_time, n_batches, args.br * n_batches)

        #         print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(args.br * n_batches) + " reviews)")
        # print()

        # trn_loader = None
        # sentiment_data = None
        

        ################################################
        # CIFAR10 RAM->GPU
        ################################################
        model_list = ["ResNet18", "ResNet152"] 

#         print("CIFAR10 benchmark (RAM->GPU data transfer)")

#         for model_type in model_list:
#             print("[" + model_type + "]")
#             for device in cuda_devices:
#                 trn_loader = make_cifar10_dataset_resnet(args.d, args.b, device=device, distributed=False, num_workers=0)
#                 model_time, n_batches = train_cnn_ram(model_type, trn_loader, device)

#                 add_item(stats, "ram_gpu", "cuda:" + str(device), 
#                          model_type, model_time, n_batches, args.b * n_batches)

#                 print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(args.b * n_batches) + " images)")
#         print()


        ################################################
        # CIFAR10 training
        ################################################
        print("CIFAR10 benchmark (full+disk)")
        for model_type in model_list:
            for num_workers in range(0, max_cpu_count):
                print("[" + model_type + " #workers ", num_workers, "]", sep="")
                for device in cuda_devices:
                    trn_loader = make_cifar10_dataset_resnet(args.d, args.b, device=device, distributed=False, num_workers=num_workers)
                    model_time, n_batches = train_cnn_full(model_type, trn_loader, device)

                    add_item(stats, "cifar" + str(num_workers).zfill(2), "cuda:" + str(device), 
                             model_type, model_time, n_batches, args.b * n_batches)

                    print("  cuda:" + str(device), model_time, "sec / batch (" + str(n_batches) + " batches, " + str(args.b * n_batches) + " images)")
        print()

    df = json_normalize(stats)
    df.sort_values(by=["model", "benchmark", "device"], inplace=True)
    print(df)
    df.to_csv(args.o + "/logs.txt")