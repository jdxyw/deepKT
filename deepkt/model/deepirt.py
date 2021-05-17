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
        self.rnn = None

        self.q_embedding = nn.Embedding(n_skill, q_embed_dim, padding_idx=n_skill+1)
        self.qa_embedding = nn.Embedding(2*n_skill+1, qa_embed_dim, padding_idx=2*n_skill)

        self.q_kp_relation = nn.Linear(self.n_skill, self.kp_dim)
        self.q_difficulty = nn.Linear(self.n_skill, self.kp_dim)
        self.user_ability = nn.Linear(self.hidden_dim, self.kp_dim)

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

        if self.rnn is None:
            raise ValueError("cell type only support lstm, rnn or gru type.")