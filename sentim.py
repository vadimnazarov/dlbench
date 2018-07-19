import os
import glob
import tarfile
import time

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch import optim


def train_sentiment(trn_loader, alphabet_size, device="cuda:0"):
    assert device != "cpu"
    if type(device) is int:
        device = "cuda:" + str(device)

    model = SentimentRNN(alphabet_size).to(device)

    optimizer = optim.RMSprop(model.parameters())

    start = time.time()
    for batch_i, (batch, lens, labels) in enumerate(trn_loader):
        batch, labels = batch.to(device), labels.to(device)
        s_values, indices = torch.sort(lens, descending=True)
        batch = torch.nn.utils.rnn.pack_padded_sequence(batch[indices], s_values, batch_first=True)
        preds = model(batch)
        loss = F.binary_cross_entropy(preds, labels[indices])
        loss.backward()
        optimizer.step()
    end = time.time()

    batch_i += 1
    return round((end - start) / batch_i, 3), batch_i


class SentimentRNN(nn.Module):

    def __init__(self, alphabet_size):
        super(SentimentRNN, self).__init__()
        hidden = 64
        self.n_layers = 2
        self.rnn = nn.GRU(alphabet_size, hidden, self.n_layers, batch_first=True)
        self.final = nn.Sequential(nn.BatchNorm1d(hidden), 
                                   nn.RReLU(), 
                                   nn.Linear(hidden, 1), 
                                   nn.Sigmoid())

    def forward(self, input):
        _, x = self.rnn(input)
        x = x[-1]
        return self.final(x.squeeze(0)).squeeze(1)


def make_imdb_dataset(data_path):
    print("IMDB: extracting files...")
    if not os.path.exists(data_path + "/imdb"):
        obj = tarfile.open(data_path + "/imdb.tgz")
        obj.extractall(data_path)
        obj.close()
    
    print("IMDB: preprocessing sequences...")
    alphabet_d = {}
    raw_seq_data = []
    lens_data = [] 
    label_data = []
    for key in ["pos", "neg"]:
        for filename in glob.glob("data/imdb/" + key + "/*"):
            with open(filename, encoding="utf-8") as f:
                content = f.read().strip().lower()
                if len(content) < 3000:
                    raw_seq_data.append(list(content))
                    for key in raw_seq_data[-1]:
                       alphabet_d[key] = alphabet_d.get(key, 0) + 1
                    lens_data.append(len(raw_seq_data[-1]))
                    label_data.append(1 if key == "pos" else 0)

    # remove low-frequency letters
    alphabet = list(filter(lambda x: alphabet_d[x] > 2000, alphabet_d))
    other_symbol = "@"
    pad_symbol = "$"
    alphabet.append(pad_symbol)
    alphabet.append(other_symbol)
    char_indices = {x:i for i,x in enumerate(sorted(alphabet))}
    get_char_index = lambda x: char_indices.get(x, char_indices[other_symbol])

    max_len = max(lens_data)

    print("IMDB: converting sequences to tensors...")
    seq_data = torch.zeros((len(raw_seq_data), max_len, len(alphabet)), dtype=torch.float)
    # seq_data = torch.zeros((500, max_len, len(alphabet)), dtype=torch.float)
    for seq_i in range(seq_data.shape[0]):
        # for symb_i in range(lens_data[seq_i]):
        #     seq_data[seq_i, symb_i, get_char_index(seq_data[seq_i][symb_i])] = 1
        seq_data[seq_i, 
                 list(range(lens_data[seq_i])),
                 [get_char_index(seq_data[seq_i][symb_i]) for symb_i in range(lens_data[seq_i])]] = 1

    lens_data = torch.IntTensor(lens_data)
    label_data = torch.FloatTensor(label_data)

    train_data = torch.utils.data.TensorDataset(seq_data, lens_data, label_data)

    return train_data, len(alphabet)


def make_imdb_dataloader(train_data, batch_size, device, num_workers=0):
    torch.cuda.device(device)
    return torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)