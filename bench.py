import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

import time
import sys


batch_size = 64
n_batches = 1000
n_features = 5000


if __name__ == "__main__":
	print("Deep Learning Benchmark")
	print()
	print(" - CUDNN?", torch.backends.cudnn.enabled)
	print(" - #devices", torch.cuda.device_count())
	# torch.cuda.set_device(args.gpu)


	layers = []
	layers.append(nn.Linear(n_features, 64))
	layers.append(nn.Linear(64, 64))
	layers.append(nn.Linear(64, 64))
	layers.append(nn.Linear(64, 1))
	model = nn.Sequential(*layers)

	optimizer = optim.Adam(model.parameters())


	start = time.time()
	for i in range(n_batches):
		batch_x = torch.FloatTensor([_ for _ in range(n_features * batch_size)]).random_(-1, 1).reshape((batch_size, n_features))
		batch_y = torch.normal(torch.FloatTensor([_ for _ in range(batch_size)]))
		pred = model(batch_x)
		loss = F.mse_loss(batch_y, pred)
		loss.backward()
		optimizer.step()
	end = time.time()

	print((end - start) / (n_batches / 10), "s / 10*batch")