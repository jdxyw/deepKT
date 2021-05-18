import sys

sys.path.insert(0, "..")

import argparse
import torch
import torch.optim
import deepkt.utils
from deepkt.data import KTDataset
from deepkt.model import DeepIRT
from deepkt.loss import DeepIRTLoss
from torch.utils.data import DataLoader
import pandas as pd
import logging
from functools import partial

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                                  collate_fn=partial(deepkt.data.collate_fn, n_skill=args.num_skill))
    test_dataloader = DataLoader(test,
                                 batch_size=args.batch_size * 2,
                                 num_workers=args.num_worker,
                                 shuffle=False,
                                 collate_fn=partial(deepkt.data.collate_fn, n_skill=args.num_skill))

    deepirt = DeepIRT(args.num_skill,
                      args.q_embed_dim,
                      args.qa_embed_dim,
                      args.hidden_dim,
                      args.kp_dim,
                      args.layer_num,
                      args.dropout,
                      device=device)
    optimizer = torch.optim.Adam(deepirt.parameters(), lr=args.learning_rate)
    loss_func = DeepIRTLoss()

    deepirt.to(device)
    loss_func.to(device)

    for epoch in range(args.epoch):
        deepkt.utils.train_epoch(deepirt, train_dataloader, optimizer, loss_func,
                                 device)
        deepkt.utils.eval_epoch(deepirt, test_dataloader, loss_func, deepkt.utils.deepirt_eval, device)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="train deep IRT model")
    arg_parser.add_argument("--learning_rate",
                            dest="learning_rate",
                            default=0.005,
                            type=float,
                            required=False)
    arg_parser.add_argument("--batch_size",
                            dest="batch_size",
                            default=64,
                            type=int,
                            required=False)
    arg_parser.add_argument("--num_skill",
                            dest="num_skill",
                            default=100,
                            type=int,
                            required=False)
    arg_parser.add_argument("--q_embed_dim",
                            dest="q_embed_dim",
                            default=64,
                            type=int,
                            required=False)
    arg_parser.add_argument("--qa_embed_dim",
                            dest="qa_embed_dim",
                            default=64,
                            type=int,
                            required=False)
    arg_parser.add_argument("--hidden_dim",
                            dest="hidden_dim",
                            default=64,
                            type=int,
                            required=False)
    arg_parser.add_argument("--kp_dim",
                            dest="kp_dim",
                            default=32,
                            type=int,
                            required=False)
    arg_parser.add_argument("--layer_num",
                            dest="layer_num",
                            default=1,
                            type=int,
                            required=False)
    arg_parser.add_argument("--dropout",
                            dest="dropout",
                            default=0.2,
                            type=float,
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