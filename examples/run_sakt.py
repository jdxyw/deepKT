import sys

sys.path.insert(0, "..")

import argparse
import torch
import torch.optim
import deepkt.utils
from deepkt.data import SAKTDataset
from deepkt.model import SAKTModel
from deepkt.loss import SAKTLoss
from torch.utils.data import DataLoader
import pandas as pd
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv("../data/assist2015_train.csv",
                           header=None,
                           sep='\t')
    test_df = pd.read_csv("../data/assist2015_test.csv", header=None, sep='\t')

    train = SAKTDataset(train_df, args.num_skill, max_len=100)
    test = SAKTDataset(test_df, args.num_skill, max_len=100)
    train_dataloader = DataLoader(train,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_worker,
                                  shuffle=True)
    test_dataloader = DataLoader(test,
                                 batch_size=args.batch_size * 2,
                                 num_workers=args.num_worker,
                                 shuffle=False)

    sakt = SAKTModel(args.num_skill, args.embed_dim, args.dropout, args.num_heads, device=device, max_len=100)

    optimizer = torch.optim.Adam(sakt.parameters(), lr=args.learning_rate)
    loss_func = SAKTLoss()

    sakt.to(device)
    loss_func.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    for epoch in range(args.epoch):
        deepkt.utils.train_epoch(sakt, train_dataloader, optimizer, loss_func,
                                 device)
        deepkt.utils.eval_epoch(sakt, test_dataloader, loss_func, deepkt.utils.sakt_eval, device)
        scheduler.step()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="train deep IRT model")
    arg_parser.add_argument("--learning_rate",
                            dest="learning_rate",
                            default=0.001,
                            type=float,
                            required=False)
    arg_parser.add_argument("--batch_size",
                            dest="batch_size",
                            default=256,
                            type=int,
                            required=False)
    arg_parser.add_argument("--num_skill",
                            dest="num_skill",
                            default=100,
                            type=int,
                            required=False)
    arg_parser.add_argument("--embed_dim",
                            dest="embed_dim",
                            default=100,
                            type=int,
                            required=False)
    arg_parser.add_argument("--dropout",
                            dest="dropout",
                            default=0.2,
                            type=float,
                            required=False)
    arg_parser.add_argument("--num_heads",
                            dest="num_heads",
                            default="4",
                            type=int,
                            required=False)
    arg_parser.add_argument("--epoch",
                            dest="epoch",
                            default=5,
                            type=int,
                            required=False)
    arg_parser.add_argument("--num_worker",
                            dest="num_worker",
                            default=0,
                            type=int,
                            required=False)
    args = arg_parser.parse_args()
    run(args)