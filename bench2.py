import time
import sys
import json
from argparse import ArgumentParser
import multiprocessing as mp

import numpy as np
from pandas.io.json import json_normalize
from tqdm import tqdm

import torch
from cnn import *

DATA = "./data"
BATCH_SIZE = 256
max_cpu_count = 4

download_time, untar_time = download_cifar10(DATA)
print("  download time:", round(download_time, 3))
print("  untar time:", round(untar_time, 3))

num_workers = 4

# trn_loader = make_cifar10_dataset_resnet(DATA, BATCH_SIZE, device="cuda", 
#                                          distributed=False, num_workers=num_workers)

model = make_model("resnet152").to("cuda")
for i in tqdm(range(100)):
    time.sleep(5)