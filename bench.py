import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

import time
import sys
from argparse import ArgumentParser

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

	return round((end - start) / (n_batches / 10), 3)


def train_cnn(trn_loader, tst_loader, device="cpu"):
	model = resnet50()
	model.to(device)

	optimizer = optim.Adam(model.parameters())

	start = time.time()
	for batch_i, (batch, labels) in enumerate(trn_loader):
		preds = model(batch)
		loss = F.cross_entropy(preds, labels)
		loss.backward()
		optimizer.step()
	end = time.time()


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

	print("CIFAR10 dataset")
	download_time, untar_time = download_cifar10(args.d)
	print(" - download time:", round(download_time, 3))
	print(" - untar time:", round(untar_time, 3))

	trn_loader, tst_loader = make_cifar10_dataset(args.d, args.b, 1024, distributed=False, num_workers=0)

	print("Simple DNN benchmark")
	model_time = train_dnn(args.b, args.n, args.f)
	print("  cpu:", model_time, "sec / 10*batch")

	if torch.cuda.device_count():
		print("[CUDNN benchmark OFF]")
		torch.backends.cudnn.benchmark = False
		for device in range(torch.cuda.device_count()):
			model_time = train_dnn(args.b, args.n, args.f, device)
			print("  cuda:" + str(device), model_time, "sec / 10*batch")
			
		if torch.backends.cudnn.enabled:
			print("[CUDNN benchmark ON]")
			torch.backends.cudnn.benchmark = True
			for device in range(torch.cuda.device_count()):
				model_time = train_dnn(args.b, args.n, args.f, device)
				print("  cuda:" + str(device), model_time, "sec / 10*batch")

			print("ResNet50 CIFAR10 benchmark")
			for device in range(torch.cuda.device_count()):
				model_time = train_cnn(trn_loader, tst_loader, device)
				print("  cuda:" + str(device), model_time, "sec / 10*batch")


