import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

import time
import sys
from argparse import ArgumentParser


def train(batch_size, n_batches, n_features, cuda_is_enabled=False):
	layers = []
	layers.append(nn.Linear(n_features, 64))
	layers.append(nn.Linear(64, 64))
	layers.append(nn.Linear(64, 64))
	layers.append(nn.Linear(64, 1))
	model = nn.Sequential(*layers)
	if cuda_is_enabled:
		model.cuda()

	optimizer = optim.Adam(model.parameters())

	start = time.time()
	for i in range(n_batches):
		batch_x = torch.FloatTensor([_ for _ in range(n_features * batch_size)]).random_(-1, 1).reshape((batch_size, n_features))
		batch_y = torch.normal(torch.FloatTensor([_ for _ in range(batch_size)]))
		if cuda_is_enabled:
			batch_x = batch_x.cuda()
			batch_y = batch_y.cuda()
		pred = model(batch_x)
		loss = F.mse_loss(batch_y, pred)
		loss.backward()
		optimizer.step()
	end = time.time()

	print((end - start) / (n_batches / 10), "s / 10*batch")


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-b", default=64, help="batch size")
	parser.add_argument("-n", default=100, help="number of batches")
	parser.add_argument("-f", default=5000, help="number of features")
	args = parser.parse_args()

	print("Deep Learning Benchmark")
	print()
	print(" - CUDA?", torch.cuda.is_available())
	print(" - CUDNN?", torch.backends.cudnn.enabled)
	print(" - #devices", torch.cuda.device_count())

	print("CPU")
	train(args.b, args.n, args.f)

	if torch.cuda.device_count():
		print("CUDNN benchmark OFF")
		torch.backends.cudnn.benchmark = False
		for device in range(torch.cuda.device_count()):
			print("GPU", device)
			train(args.b, args.n, args.f, torch.cuda.is_available())

		if torch.backends.cudnn.enabled:
			print("CUDNN benchmark ON")
			torch.backends.cudnn.benchmark = True
			for device in range(torch.cuda.device_count()):
				print("GPU", device)
				train(args.b, args.n, args.f, torch.cuda.is_available())