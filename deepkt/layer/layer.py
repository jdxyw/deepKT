import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, state_size=200, dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = state_size
        self.dropout = dropout
        self.lr1 = nn.Linear(self.state_size, self.state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(self.state_size, self.state_size)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)
