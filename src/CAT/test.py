#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from cat import Cat

sys.path.append('.')
from src.utils.dataset_utils import read_dataset
from src.utils.db_utils import DBConn
from src.utils.sampler_utils import ItemwiseSampler
from src.utils.path_utils import get_stem

class PlanDataset(Dataset):
    def __init__(self, dataset: list[list[dict]], model: Cat):
        plans = [plan for query in tqdm(dataset) for plan in query]
        self.model = model
        self.samples = model.transform(plans)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def test(model, dataloader):
        model.model.cuda()
        model.model.eval()
        preds = []
        for x, pos, mask, _, _ in tqdm(dataloader):
            cost = model.model.cost_output(x, pos, mask)
            pred = cost.view(-1)
            preds.append(pred.detach().cpu().numpy())
        preds = np.concatenate(preds)
        return preds

def group_by_dataset(costs: np.ndarray, dataset: list[list[dict]]):
    dataset_costs = []
    idx = 0
    for query in dataset:
        dataset_costs.append(costs[idx:idx+len(query)])
        idx += len(query)
    return dataset_costs

def best_preds(costs: list[np.ndarray]) -> list[int]:
    return [np.argmin(cost).item() for cost in costs]

def main(args: argparse.Namespace):
    dataset_regex = re.compile(r'([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)')
    database, workload, _ = dataset_regex.match(args.dataset).groups()

    # read dataset
    dataset_path = os.path.join('datasets', args.dataset)
    _, dataset = read_dataset(dataset_path)

    # read db info
    with DBConn(database) as db:
        db_info = db.get_db_info()

    # create model
    model = Cat(db_info)
    model.load(args.model)

    test_dataset = PlanDataset(dataset, model)
    itemwise_sampler = ItemwiseSampler(dataset, args.batch_size, shuffle=False)
    dataloader = DataLoader(test_dataset, batch_sampler=itemwise_sampler, collate_fn=model.get_collate_fn(torch.device('cuda')))
    results = test(model, dataloader)
    results = group_by_dataset(results, dataset)
    results = best_preds(results)
    result_dir = os.path.join('results', args.dataset)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f'{get_stem(args.model)}.json')
    with open(result_path, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb/JOB-sample/JOP')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model', type=str, default='cost.pth')
    args = parser.parse_args()
    main(args)