# -*- coding:utf-8 -*-
"""
    Paper reference: Deep-IRT: Make Deep Learning Based Knowledge Tracin
    Explainable Using Item Response Theory (https://arxiv.org/pdf/1904.11738.pdf)

    Note:
        This DeepIRT is different from the original paper. It's a variant of the IRT.
        The model here use a IRT to do the prediction, and use a sequence model to
        estimate user's ability. For alpha and beta of IRT, each one has its own network
        to do the estimation for each question/skill.
"""
import torch
import torch.nn as nn


class DeepIRT(nn.Module):
    def __init__(
        self,
        n_skill,
        q_embed_dim,
        qa_embed_dim,
        hidden_dim,
        kp_dim,
        n_layer,
        dropout,
        device="cpu",
        cell_type="lstm",
    ):
        super(DeepIRT, self).__init__()
        self.n_skill = n_skill
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.hidden_dim = hidden_dim
        self.kp_dim = kp_dim
        self.n_layer = n_layer
        self.dropout = dropout
        self.device = device
        self.rnn = None

        self.q_embedding = nn.Embedding(n_skill+1, q_embed_dim, padding_idx=n_skill)
        self.qa_embedding = nn.Embedding(2*n_skill+1, qa_embed_dim, padding_idx=2*n_skill)

        self.q_kp_relation = nn.Linear(self.q_embed_dim, self.kp_dim)
        self.q_difficulty = nn.Linear(self.q_embed_dim, self.kp_dim)
        self.user_ability = nn.Linear(self.hidden_dim, self.kp_dim)

        if cell_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                self.qa_embed_dim,
                self.hidden_dim,
                self.n_layer,
                batch_first=True,
                dropout=self.dropout,
            )
        elif cell_type.lower() == "rnn":
            self.rnn = nn.RNN(
                self.qa_embed_dim,
                self.hidden_dim,
                self.n_layer,
                batch_first=True,
                dropout=self.dropout,
            )
        elif cell_type.lower() == "gru":
            self.rnn = nn.GRU(
                self.qa_embed_dim,
                self.hidden_dim,
                self.n_layer,
                batch_first=True,
                dropout=self.dropout,
            )

        if self.rnn is None:
            raise ValueError("cell type only support lstm, rnn or gru type.")

    def forward(self, q, qa):
        q_embed_data = self.q_embedding(q)
        qa_embed_data = self.qa_embedding(qa)

        batch_size = q.size(0)
        seq_len = q.size(1)

        # h0 = torch.zeros((q.size(0), self.n_layer, self.hidden_dim), device=self.device)
        states, _ = self.rnn(qa_embed_data)
        # states_before = torch.cat((h0, states[:, :-1, :]), 1)
        user_ability = self.user_ability(states).view(batch_size*seq_len, -1)

        kp_relation = torch.softmax(self.q_kp_relation(q_embed_data.view(batch_size*seq_len, -1)), dim=1)
        item_difficulty = self.q_difficulty(q_embed_data.view(batch_size*seq_len, -1))

        logits = (user_ability - item_difficulty) * kp_relation
        return logits.sum(dim=1), None