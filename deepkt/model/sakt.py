# -*- coding:utf-8 -*-
"""
    Reference: A Self-Attentive model for Knowledge Tracing (https://arxiv.org/abs/1907.06837)
"""

import torch
import torch.nn as nn
import deepkt.utils
import deepkt.layer


class SAKTModel(nn.Module):
    def __init__(
        self, n_skill, embed_dim, dropout, num_heads=4, max_len=64, device="cpu"
    ):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.q_embed_dim = embed_dim
        self.qa_embed_dim = embed_dim
        self.pos_embed_dim = embed_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.max_len = max_len
        self.device = device

        self.q_embedding = nn.Embedding(
            n_skill + 1, self.q_embed_dim, padding_idx=n_skill
        )
        self.qa_embedding = nn.Embedding(
            2 * n_skill + 2, self.qa_embed_dim, padding_idx=2 * n_skill + 1
        )
        self.pos_embedding = nn.Embedding(self.max_len, self.pos_embed_dim)

        self.multi_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout
        )

        self.key_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.query_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.ffn = deepkt.layer.FFN(self.embed_dim)
        self.pred = nn.Linear(self.embed_dim, 1, bias=True)

    def forward(self, q, qa):
        qa = self.qa_embedding(qa)
        pos_id = torch.arange(qa.size(1)).unsqueeze(0).to(self.device)
        pos_x = self.pos_embedding(pos_id)
        qa = qa + pos_x
        q = self.q_embedding(q)

        q = q.permute(1, 0, 2)
        qa = qa.permute(1, 0, 2)

        attention_mask = deepkt.utils.future_mask(q.size(0)).to(self.device)
        attention_out, _ = self.multi_attention(q, qa, qa, attn_mask=attention_mask)
        attention_out = self.layer_norm1(attention_out + q)
        attention_out = attention_out.permute(1, 0, 2)

        x = self.ffn(attention_out)
        x = self.dropout_layer(x)
        x = self.layer_norm2(x + attention_out)
        x = self.pred(x)

        return x.squeeze(-1), None
