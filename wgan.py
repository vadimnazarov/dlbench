#
# Modified code from https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
#

import os
import time

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim

import torch.autograd as autograd


def train_wgan(trn_loader, device="cuda:0"):
    # assert device != "cpu"
    if type(device) is int:
        device = "cuda:" + str(device)

    model_Gen = Generator().to(device)
    model_Dis = Discriminator().to(device)

    optim_Gen = optim.Adam(model_Gen.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optim_Dis = optim.Adam(model_Dis.parameters(), lr=1e-4, betas=(0.5, 0.9))

    max_iterations = 100
    crit_iters = 5
    lambda_val = 10

    start = time.time()
    for iteration in range(max_iterations):
        for batch_i, (batch, _) in enumerate(trn_loader):
            optim_Dis.zero_grad()
            batch = batch.to(device)

            D_real = -model_Dis(batch).mean()
            D_real.backward()

            noise = torch.randn(batch.size(0), DIM, device=device)
            model_Gen.eval()
            fake = model_Gen(noise)
            D_fake = model_Dis(fake).mean()
            D_fake.backward(retain_graph=True)

            gradient_penalty = calc_gradient_penalty(model_Dis, batch, fake, lambda_val, device)
            gradient_penalty.backward()
            model_Gen.train()

            D_cost = D_fake + D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optim_Dis.step()

            if batch_i == crit_iters:
                break 

        model_Dis.eval()
        optim_Gen.zero_grad()

        noise = torch.randn(batch.size(0), DIM, device=device)
        fake = model_Gen(noise)
        G_loss = -model_Dis(fake).mean()
        G_loss.backward()
        optim_Gen.step()
        model_Dis.train()

    end = time.time()

    batch_i = (max_iterations * crit_iters)
    return round((end - start) / batch_i, 3), batch_i


DIM = 64
OUTPUT_DIM = 3072


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(DIM, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output


def calc_gradient_penalty(model_Dis, real_data, fake_data, lambda_val, device):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size(0), real_data.nelement() / real_data.size(0)).contiguous().view(real_data.size(0), 3, 32, 32)
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)

    disc_interpolates = model_Dis(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_val
    return gradient_penalty


def make_cifar10_dataset_wgan(data_path, batch_size, device, num_workers=0):
    torch.cuda.device(device)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_folder = os.path.join(data_path, 'train')
    train_data = torchvision.datasets.ImageFolder(train_folder, transform_train)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return trainloader