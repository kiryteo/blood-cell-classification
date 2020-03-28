
import torch
import torch.nn as nn
from typing import *


import numpy as np


class RecurrentConvNet(nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()

        self.device = device
        self.hidden = 256

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1,8, kernel_size=8, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        )
        
        self.rnnCell = nn.LSTMCell(10816, self.hidden)
        self.rnn = nn.LSTM(self.hidden, self.hidden)

        self.classifier = nn.Sequential(nn.Linear(self.hidden, 32), nn.Tanh(), nn.Linear(32, 3))


    def forward(self, x):
        # We assume x = (seq_length, batch, 31, 31)
        x = torch.unsqueeze(x, dim=2)
        # x = (seq_length, batch, 1, 31, 31)

        batch_size = x.shape[1]
        seq_length = x.shape[0]

        hidden = torch.zeros((batch_size, self.hidden)).to(self.device)
        cell = torch.zeros((batch_size, self.hidden)).to(self.device)

        hiddens = [hidden]
        for i in range(seq_length):
            features = self.feature_extractor(x[i])
            # features = (batch, 64, h, w)

            features = torch.flatten(features, start_dim=1, end_dim=-1)

            hidden, cell = self.rnnCell(features, (hiddens[-1], cell))
            hiddens.append(hidden)

        output, (h, c) = self.rnn(torch.stack(hiddens))

        return nn.functional.log_softmax(self.classifier(output[-1]), dim=1)









