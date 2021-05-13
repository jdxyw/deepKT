import torch
import deepkt.loss
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import numpy as np
import os
import random

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, train_iterator, optim, criterion, device="cpu"):
    model.train()

    for i, (q, qid, labels, mask) in enumerate(train_iterator):
        q, qid, labels, mask = q.to(device), qid.to(device), labels.to(
            device), mask.to(device)

        optim.zero_grad()
        logits, _ = model(q)
        loss = criterion(logits, labels, qid, mask, device=device)
        loss.backward()
        optim.step()


def eval_epoch(model, test_iterator, criterion, device="cpu"):
    model.eval()

    eval_loss = []
    preds, binary_preds, targets = [], [], []
    for i, (q, qid, labels, mask) in enumerate(test_iterator):
        q, qid, labels, mask = q.to(device), qid.to(device), labels.to(
            device), mask.to(device)

        with torch.no_grad():
            logits, _ = model(q)

        loss = criterion(logits, labels, qid, mask, device=device)
        eval_loss.append(loss.detach().item())

        mask = mask.eq(1)

        pred, binary_pred = deepkt.loss.dkt_predict(logits, qid)
        pred = torch.masked_select(pred, mask).detach().numpy()
        binary_pred = torch.masked_select(binary_pred, mask).detach().numpy()
        target = torch.masked_select(labels, mask).detach().numpy()
        # pred = pred.cpu().detach().numpy().reshape(-1)
        # binary_pred = binary_pred.cpu().detach().numpy().reshape(-1)

        preds.append(pred)
        binary_preds.append(binary_pred)
        targets.append(target)

    preds = np.concatenate(preds)
    binary_preds = np.concatenate(binary_preds)
    targets = np.concatenate(targets)

    auc_value = roc_auc_score(targets, preds)
    accuracy = accuracy_score(targets, binary_preds)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        targets, binary_preds)
    pos_rate = np.sum(targets) / float(len(targets))
    print(
        "auc={0}, accuracy={1}, precision={2}, recall={3}, fscore={4}, pos_rate={5}"
        .format(auc_value, accuracy, precision, recall, f_score, pos_rate))
