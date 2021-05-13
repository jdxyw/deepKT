# -*- coding:utf-8 -*-
import torch.utils.data
import torch.nn.utils
import numpy as np


class KTDataset(torch.utils.data.Dataset):
    def __init__(self, df, n_skill):
        super(KTDataset, self).__init__()
        self.df = df
        self.n_skill = n_skill

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        qids = self.df[0][idx].split(",")
        correct = self.df[1][idx].split(",")

        qids = list(map(int, qids))
        correct = list(map(int, correct))

        qid, qa = torch.LongTensor(qids), torch.LongTensor(correct)
        qa = qid + qa * self.n_skill

        mask = np.ones((len(qid),), dtype=int)

        return torch.cat((torch.LongTensor([2 * self.n_skill]), qa[:-1])), \
               qid.view(-1, 1), torch.LongTensor(correct), torch.LongTensor(mask)


def collate_fn(data):
    q = [x[0] for x in data]
    qid = [x[1] for x in data]
    qc = [x[2] for x in data]
    mask = [x[3] for x in data]
    q = torch.nn.utils.rnn.pad_sequence(q, batch_first=True, padding_value=0)
    qid = torch.nn.utils.rnn.pad_sequence(qid,
                                          batch_first=True,
                                          padding_value=0)
    qc = torch.nn.utils.rnn.pad_sequence(qc, batch_first=True, padding_value=-1)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)

    return q, qid, qc, mask
