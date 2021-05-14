# deepKT

Knowledge tracing(**KT**) is the crucial task in the filed of online education. Before the rise of the deep learning, **IRT**, **BKT** and other traditional models are used to predict users/students ability and proficiency. 

With the development of the deep learning, the **DL** also shows its power in this field. **KT** is a well-defined task. In short, it would give the probability for the correctness of next question which student need solve.

This repo would implement some Deep Knowledge Tracing models with PyTorch.

## Data

Under the folder `data`, this repo provides `ASSISTments2015` dataset. This dataset has been processed, not the original official format. This dataset is the simplest dataset that only contains the correct tag of an attempt. If your dataset is more complex with other information, you could define your PyTorch `Dataset`.

The data format.

```text
45,45,45,47,47,47,28,28,28,28,28,17,17,17,28,28,28	1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1
19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19	0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1
49,49,49,49,49,49,49,92,92,92,92,92,26,26,26,26,26,26	0,1,0,0,1,1,1,0,0,1,1,1,1,0,0,1,1,1
```

Each line contains two fields separated by `\t`. The first field is the `question id` sequence answered by the user. The second filed is the corresponding answer sequence. `1` means correct, `0` means wrong.

## Project structure

```
├── data
│   ├── assist2015_test.csv
│   └── assist2015_train.csv
├── deepkt
│   ├── data
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── loss
│   ├── model
│   └── utils
│       ├── __init__.py
│       └── utils.py
└── examples
```

| Folder       | Usage                          |
| ------------ | ------------------------------ |
| data         | example data                   |
| deepkt/data  | define PyTorch Dataset         |
| deepkt/loss  | define KT loss                 |
| deepkt/model | define KT model                |
| deepkt/utils | define some utils              |
| examples     | each model has its own example |

## References

| Model                       | Paper                                                                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Deep Knowledge Tracing      | [Deep Knowledge Tracing](https://arxiv.org/abs/1506.05908)                                                                         |
| Deep Knowledge Tracing Plus | [Addressing Two Problems in Deep Knowledge Tracing via Prediction-Consistent Regularization](https://arxiv.org/pdf/1806.02180.pdf) |