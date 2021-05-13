import sys

sys.path.insert(0, "..")

import argparse
import torch
import torch.optim
import deepkt.utils
from deepkt.data import KTDataset
from deepkt.model import DKT
from deepkt.loss import DKTLoss
from torch.utils.data import DataLoader
import pandas as pd
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    train_df = pd.read_csv("../data/assist2015_train.csv",
                           header=None,
                           sep='\t')
    test_df = pd.read_csv("../data/assist2015_test.csv", header=None, sep='\t')

    train = KTDataset(train_df, args.num_skill)
    test = KTDataset(test_df, args.num_skill)
    train_dataloader = DataLoader(train,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_worker,
                                  shuffle=True,
                                  collate_fn=deepkt.data.collate_fn)
    test_dataloader = DataLoader(test,
                                 batch_size=args.batch_size * 2,
                                 num_workers=args.num_worker,
                                 shuffle=False,
                                 collate_fn=deepkt.data.collate_fn)

    dkt = DKT(args.embed_dim,
              args.num_skill * 2 + 1,
              args.hidden_dim,
              args.layer_num,
              args.num_skill,
              device=device)
    optimizer = torch.optim.Adam(dkt.parameters(), lr=args.learning_rate)
    loss_func = DKTLoss()

    dkt.to(device)
    loss_func.to(device)

    for epoch in range(args.epoch):
        deepkt.utils.train_epoch(dkt, train_dataloader, optimizer, loss_func,
                                 device)
        deepkt.utils.eval_epoch(dkt, test_dataloader, loss_func, device)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="train dkt model")
    arg_parser.add_argument("--learning_rate",
                            dest="learning_rate",
                            default=0.001,
                            type=float,
                            required=False)
    arg_parser.add_argument("--batch_size",
                            dest="batch_size",
                            default=32,
                            type=int,
                            required=False)
    arg_parser.add_argument("--num_skill",
                            dest="num_skill",
                            default=100,
                            type=int,
                            required=False)
    arg_parser.add_argument("--embed_dim",
                            dest="embed_dim",
                            default=32,
                            type=int,
                            required=False)
    arg_parser.add_argument("--hidden_dim",
                            dest="hidden_dim",
                            default=128,
                            type=int,
                            required=False)
    arg_parser.add_argument("--layer_num",
                            dest="layer_num",
                            default=1,
                            type=int,
                            required=False)
    arg_parser.add_argument("--output_dim",
                            dest="output_dim",
                            default=100,
                            type=int,
                            required=False)
    arg_parser.add_argument("--cell_type",
                            dest="cell_type",
                            default="lstm",
                            type=str,
                            required=False)
    arg_parser.add_argument("--epoch",
                            dest="epoch",
                            default=10,
                            type=int,
                            required=False)
    arg_parser.add_argument("--num_worker",
                            dest="num_worker",
                            default=0,
                            type=int,
                            required=False)
    args = arg_parser.parse_args()
    run(args)
