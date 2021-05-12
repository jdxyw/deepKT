# -*- coding:utf-8 -*-
"""
    Paper reference: Deep Knowledge Tracing (https://arxiv.org/abs/1506.05908)
"""

import torch
import torch.nn as nn


class DKT(nn.Module):
    def __init__(self,
                 embed_dim,
                 input_dim,
                 hidden_dim,
                 layer_num,
                 output_dim,
                 device="cpu",
                 cell_type="lstm"):
        super(DKT, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.output_dim = output_dim
        self.device = device
        self.cell_type = cell_type
        self.rnn = None

        self.skill_embedding = nn.Embedding(self.input_dim, self.embed_dim)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        if cell_type.lower() == "lstm":
            self.rnn = nn.LSTM(self.embed_dim,
                               self.hidden_dim,
                               self.layer_num,
                               batch_first=True)
        elif cell_type.lower() == "rnn":
            self.rnn = nn.RNN(self.embed_dim,
                              self.hidden_dim,
                              self.layer_num,
                              batch_first=True)
        elif cell_type.lower() == "gru":
            self.rnn = nn.GRU(self.embed_dim,
                              self.hidden_dim,
                              self.layer_num,
                              batch_first=True)

        if self.rnn is None:
            raise ValueError("cell type only support lstm, rnn or gru type.")

    def forward(self, x, state_in=None):
        h0 = torch.zeros((self.layer_num, x.size(0), self.hidden_dim),
                         device=self.device)
        c0 = torch.zeros((self.layer_num, x.size(0), self.hidden_dim),
                         device=self.device)

        if state_in is None:
            state_in = (h0, c0)

        state, state_out = self.rnn(x, state_in)
        logits = self.fc(state)
        return logits, state_out
