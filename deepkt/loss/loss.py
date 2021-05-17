# -*- coding:utf-8 -*-

import torch
import torch.nn as nn


class DKTLoss(nn.Module):
    def __init__(self, reduce=None):
        super(DKTLoss, self).__init__()
        self.reduce = reduce

    def forward(self, logits, targets, qid, mask, device="cpu"):
        preds = torch.sigmoid(logits)
        preds = torch.gather(preds, dim=2, index=qid)
        preds = torch.squeeze(preds)
        ones = torch.ones(targets.size(), device=device)
        total = torch.sum(mask) + 1
        loss = -torch.sum(
            mask * targets * torch.log(preds)
            + mask * (ones - targets) * torch.log(ones - preds)
        )

        if self.reduce is None or self.reduce == "mean":
            loss = loss / total

        if self.reduce is not None and self.reduce not in ["mean", "sum"]:
            raise ValueError("the reduce should be mean or sum")

        return loss


class DKTPlusLoss(nn.Module):
    def __init__(self, gamma=0.1, reg1=0.03, reg2=1.0, reduce=None):
        super(DKTPlusLoss, self).__init__()
        self.gamma = gamma
        self.reg1 = reg1
        self.reg2 = reg2
        self.reduce = reduce

    def forward(self, logits, targets, qid, mask, device="cpu"):
        preds = torch.sigmoid(logits)
        preds = torch.gather(preds, dim=2, index=qid)
        preds = torch.squeeze(preds)
        ones = torch.ones(targets.size(), device=device)
        total = torch.sum(mask) + 1
        loss1 = -torch.sum(
            mask * targets * torch.log(preds)
            + mask * (ones - targets) * torch.log(ones - preds)
        )

        loss2 = -torch.sum(
            mask[:, :-1] * targets[:, :-1] * torch.log(preds[:, 1:])
            + mask[:, :-1]
            * (ones[:, 1:] - targets[:, :-1])
            * torch.log(ones[:, 1:] - preds[:, 1:])
        )

        if self.reduce is None or self.reduce == "mean":
            loss1 = loss1 / total
            loss2 = loss2 / total

        if self.reduce is not None and self.reduce not in ["mean", "sum"]:
            raise ValueError("the reduce should be mean or sum")

        preds0 = preds[:, :-1].clone()
        preds1 = preds[:, 1:]
        preds0[preds1 == 0] = 0

        reg1 = torch.sum(torch.abs(preds1 - preds0))
        reg2 = torch.sqrt(torch.sum(torch.pow(preds1 - preds0, 2)))

        return loss1 + self.gamma * loss2 + self.reg1 * reg1 + self.reg2 * reg2


def dkt_predict(logits, qid):
    preds = torch.sigmoid(logits)
    preds = torch.gather(preds, dim=2, index=qid)
    preds = torch.squeeze(preds)
    binary_preds = torch.round(preds)
    return (
        preds.view(preds.size()[0], preds.size()[1]),
        binary_preds.view(preds.size()[0], preds.size()[1]),
    )
