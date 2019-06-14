import time
import sys

import numpy as np
from pandas.io.json import json_normalize
from tqdm import tqdm_notebook

import torch
import torch.optim as optim
import torch.nn.functional as F
from cnn import make_cifar10_dataset_resnet, make_model, download_cifar10


DATA = "./data"
BATCH_SIZE = 128
max_cpu_count = 4
do_transform = False


def add_item(stats, bench, do_transform, device, workers, 
             model, time, batches, images):
    stats.append({})
    stats[-1]["benchmark"] = bench
    stats[-1]["transform"] = do_transform
    stats[-1]["workers"] = workers
    stats[-1]["device"] = device
    stats[-1]["model"] = model
    stats[-1]["time"] = time
    stats[-1]["batches"] = batches
    stats[-1]["images/sec"] = images


if __name__ == "__main__":
    stats = []
    
    download_time, untar_time = download_cifar10(DATA)
    print("  download time:", round(download_time, 3))
    print("  untar time:", round(untar_time, 3))
    
    cuda_devices = list(range(torch.cuda.device_count()))
    model_list = ["resnet18", "resnet152"]

    for cuda_device in cuda_devices:
        for model_name in model_list:
            for num_workers in range(5):
                print("Device:\t", cuda_device)
                print("Model:\t", model_name)
                print("Workers:\t", num_workers)
                trn_loader = make_cifar10_dataset_resnet(DATA, BATCH_SIZE, device=cuda_device, transformations=do_transform,
                                                         distributed=False, num_workers=num_workers)

                model = make_model(model_name).to(cuda_device)
                optimizer = optim.Adam(model.parameters())

                start = time.time()
                for batch_i, (batch, labels) in enumerate(trn_loader):
                    optimizer.zero_grad()
                    batch, labels = batch.to(cuda_device), labels.to(cuda_device)
                    preds = model(batch)
                    loss = F.cross_entropy(preds, labels)
                    loss.backward()
                    optimizer.step()
                model_time = time.time() - start
                add_item(stats, "cifar", do_transform, cuda_device, num_workers, 
                         model_name, model_time, batch_i, BATCH_SIZE * batch_i / model_time)
                print()
                
    df = json_normalize(stats)
    print(df)
    df.to_csv("./logs.txt")