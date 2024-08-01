#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import re
import sys
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

sys.path.append('.')
from src.LeroMamba.lero import Lero
from src.utils.dataset_utils import read_dataset
from src.utils.db_utils import DBConn
from src.utils.sampler_utils import PairwiseSampler, QuerySampler

class PlanDataset(Dataset):
    def __init__(self, dataset: list[list[dict]], model: Lero):
        plans = [plan for query in dataset for plan in query]
        self.model = model
        self.samples = model.transform(plans)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def tailr_loss_with_logits(input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor|None = None, gamma: float = 0.1):
    pos = -torch.log(gamma + (1 - gamma) * torch.sigmoid(input))
    neg = -torch.log(gamma + (1 - gamma) * (1 - torch.sigmoid(input)))
    if weight is not None:
        return torch.sum(weight * (target * pos + (1 - target) * neg)) / (1 - gamma)
    else:
        return torch.mean(target * pos + (1 - target) * neg) / (1 - gamma)

def test_impl(model, test_dataloader, topk):
    model.model.cuda()
    model.model.eval()
    selected = []
    overhead = 0
    for trees, cost_label, weights in tqdm(test_dataloader):
        start = time()
        cost = model.model(trees.to('cuda'))
        end = time()
        overhead += end - start
        pred = cost.view(-1)
        if topk == 1:
            argmin_pred = torch.argmin(pred)
            selected.append([argmin_pred.item()])
        else:
            _, sorted_indices = pred.sort()
            argmin_pred = sorted_indices[0]
            selected.append(sorted_indices[:topk].tolist())
    print(overhead)
    return selected

def test(model_path, database, test, topk):
    if type(test) == str:
        dataset_path = os.path.join('datasets', test)
        names, test_queries = read_dataset(dataset_path)
    else:
        test_queries = test
    with DBConn(database) as db:
        db_info = db.get_db_info()
    model = Lero(db_info.table_map)
    model.load(model_path)
    test_dataset = PlanDataset(test_queries, model)
    test_sampler = QuerySampler(test_queries)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=model._transform_samples)
    return test_impl(model, test_dataloader, topk)

def main(args: argparse.Namespace):
    dataset_regex = re.compile(r'([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)')
    database, workload, method = dataset_regex.match(args.test).groups()
    print(test(args.model, database, args.test, args.topk))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--test', type=str, default='')
    parser.add_argument('--topk', type=int, default=1)
    args = parser.parse_args()
    main(args)