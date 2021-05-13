# -*- coding:utf-8 -*-

import torch
import torch.nn as nn


class DKTLoss(nn.Module):
    def __init__(self):
        super(DKTLoss, self).__init__()

    def forward(self, logits, targets, qid, mask, device="cpu"):
        preds = torch.sigmoid(logits)
        preds = torch.gather(preds, dim=2, index=qid)
        preds = torch.squeeze(preds)
        ones = torch.ones(targets.size(), device=device)
        loss = -torch.sum(mask * targets * torch.log(preds) + mask *
                          (ones - targets) * torch.log(ones - preds))
        return loss


def dkt_predict(logits, qid):
    preds = torch.sigmoid(logits)
    preds = torch.gather(preds, dim=2, index=qid)
    preds = torch.squeeze(preds)
    binary_preds = torch.round(preds)
    return preds.view(preds.size()[0],
                      preds.size()[1]), binary_preds.view(
                          preds.size()[0],
                          preds.size()[1])
