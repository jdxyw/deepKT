# -*- coding:utf-8 -*-
"""
    Paper reference: Deep Knowledge Tracing (https://arxiv.org/abs/1506.05908)
"""

import torch
import torch.nn as nn


class DKT(nn.Module):
    def __init__(
        self,
        embed_dim,
        input_dim,
        hidden_dim,
        layer_num,
        output_dim,
        dropout,
        device="cpu",
        cell_type="lstm",
    ):
        """ The first deep knowledge tracing network architecture.

        :param embed_dim: int, the embedding dim for each skill.
        :param input_dim: int, the number of skill(question) * 2.
        :param hidden_dim: int, the number of hidden state dim.
        :param layer_num: int, the layer number of the sequence number.
        :param output_dim: int, the number of skill(question).
        :param device: str, 'cpu' or 'cuda:0', the default value is 'cpu'.
        :param cell_type: str, the sequence model type, it should be 'lstm', 'rnn' or 'gru'.
        """
        super(DKT, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.output_dim = output_dim + 1
        self.dropout = dropout
        self.device = device
        self.cell_type = cell_type
        self.rnn = None

        self.skill_embedding = nn.Embedding(
            self.input_dim, self.embed_dim, padding_idx=self.input_dim - 1
        )

        if cell_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                self.embed_dim,
                self.hidden_dim,
                self.layer_num,
                batch_first=True,
                dropout=self.dropout,
            )
        elif cell_type.lower() == "rnn":
            self.rnn = nn.RNN(
                self.embed_dim,
                self.hidden_dim,
                self.layer_num,
                batch_first=True,
                dropout=self.dropout,
            )
        elif cell_type.lower() == "gru":
            self.rnn = nn.GRU(
                self.embed_dim,
                self.hidden_dim,
                self.layer_num,
                batch_first=True,
                dropout=self.dropout,
            )
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        if self.rnn is None:
            raise ValueError("cell type only support lstm, rnn or gru type.")

    def forward(self, q, qa, state_in=None):
        """

        :param x: The input is a tensor(int64) with 2 dimension, like [H, k]. H is the batch size,
        k is the length of user's skill/question id sequence.
        :param state_in: optional. The state tensor for sequence model.
        :return:
        """
        qa = self.skill_embedding(qa)
        h0 = torch.zeros(
            (self.layer_num, qa.size(0), self.hidden_dim), device=self.device
        )
        c0 = torch.zeros(
            (self.layer_num, qa.size(0), self.hidden_dim), device=self.device
        )

        if state_in is None:
            state_in = (h0, c0)

        state, state_out = self.rnn(qa, state_in)
        logits = self.fc(state)
        return logits, state_out
