# -*- coding:utf-8 -*-
import torch.utils.data
import torch.nn.utils


class KTDataset(torch.utils.data.Dataset):
    def __init__(self, df, n_skill):
        self.df = df
        self.n_skill = n_skill

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        qids = self.df[0][idx].split(",")
        correct = self.df[1][idx].split(",")

        q, qa = torch.LongTensor(qids), torch.LongTensor(correct)
        qa = q + qa * self.n_skill

        return q, torch.cat(
            (torch.LongTensor([2 * self.n_skill + 1]), qa[:-1]))


def collate_fn(data):
    q = [x[0] for x in data]
    qa = [x[1] for x in data]
    q = torch.nn.utils.rnn.pad_sequence(q, batch_first=True, padding_value=-1)
    qa = torch.nn.utils.rnn.pad_sequence(qa,
                                         batch_first=True,
                                         padding_value=-1)

    return q, qa
