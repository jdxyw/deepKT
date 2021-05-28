# -*- coding:utf-8 -*-
"""
    Reference: Towards an Appropriate Query, Key, and Value
    Computation for Knowledge Tracing (https://arxiv.org/pdf/2002.07033.pdf)
"""

import torch
import torch.nn as nn
import deepkt.layer
import deepkt.utils


class SaintEncoder(nn.Module):
    def __init__(self, embed_dim, dropout=0.3, num_heads=4):
        super(SaintEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.ffn = deepkt.layer.FFN(self.embed_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        device = x.device
        x = self.layer_norm1(x)
        encoder, _ = self.attn(
            x, x, x, attn_mask=deepkt.utils.future_mask(x.size(0)).to(device)
        )
        encoder = encoder + x

        encoder = encoder.permute(1, 0, 2)
        encoder_out = self.layer_norm2(encoder)

        return self.ffn(encoder_out) + encoder_out


class SaintDecoder(nn.Module):
    def __init__(self, embed_dim, dropout=0.3, num_heads=4):
        super(SaintDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_heads = num_heads

        self.attn1 = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout
        )
        self.attn2 = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout
        )

        self.ffn = deepkt.layer.FFN(self.embed_dim)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)

    def forward(self, decoder_in, encoder_out):
        device = decoder_in.device
        x = self.layer_norm1(decoder_in)
        decoder, _ = self.attn1(
            x, x, x, attn_mask=deepkt.utils.future_mask(x.size(0)).to(device)
        )
        decoder = decoder + x

        encoder_out = encoder_out.permute(1, 0, 2)
        encoder_out = self.layer_norm2(encoder_out)
        encoder_out2 = self.attn2(decoder, encoder_out, encoder_out)[0]
        decoder_out = decoder + encoder_out2

        decoder_out = decoder_out.permute(1, 0, 2)
        decoder_out = self.layer_norm3(decoder_out)
        return decoder_out + self.ffn(decoder_out)


class SaintModel(nn.Module):
    def __init__(
        self,
        n_skill,
        embed_dim,
        dropout,
        num_heads=4,
        num_enc=4,
        max_len=64,
        device="cpu",
    ):
        super(SaintModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_enc = num_enc
        self.max_len = max_len
        self.device = device

        self.q_embedding = nn.Embedding(
            n_skill + 1, self.embed_dim, padding_idx=n_skill
        )
        self.qa_embedding = nn.Embedding(
            2 * n_skill + 2, self.embed_dim, padding_idx=2 * n_skill + 1
        )
        self.pos_embedding = nn.Embedding(self.max_len, self.embed_dim)

        self.encoders = nn.ModuleList(
            [
                SaintEncoder(self.embed_dim, self.dropout, self.num_heads)
                for x in range(self.num_enc)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                SaintDecoder(self.embed_dim, self.dropout, self.num_heads)
                for x in range(self.num_enc)
            ]
        )

        self.fc = nn.Linear(self.embed_dim, 1)

    def forward(self, q, qa):
        qa = self.qa_embedding(qa)
        pos_id = torch.arange(qa.size(1)).unsqueeze(0).to(self.device)
        pos_x = self.pos_embedding(pos_id)
        qa = qa + pos_x
        q = self.q_embedding(q)

        q = q.permute(1, 0, 2)
        qa = qa.permute(1, 0, 2)

        for x in range(self.num_enc):
            q = self.encoders[x](q)

        for x in range(self.num_enc):
            qa = self.decoders[x](qa, q)

        logits = self.fc(qa)
        return logits, None
