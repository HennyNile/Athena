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

from cat2 import Cat

sys.path.append('.')
from src.utils.dataset_utils import read_dataset
from src.utils.db_utils import DBConn
from src.utils.sampler_utils import BatchedQuerySampler
from src.utils.path_utils import get_stem

class PlanDataset(Dataset):
    def __init__(self, dataset: list[list[dict]], model: Cat):
        def get_num_finished(query: list[dict]):
            return sum([1 if 'Execution Time' in plan else 0 for plan in query])
        plans = [plan for query in dataset for plan in query]
        num_finished = [max(1, get_num_finished(query)) for query in dataset for _ in query]
        self.samples = model.transform(plans, num_finished)
        print("Preparing input tensors:")
        self.inputs = [model.transform_sample(sample) for sample in tqdm(self.samples)]
        self.model = model
        self.num_samples = len(self.samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx]

def test(model, dataloader):
        model.model.cuda()
        model.model.eval()
        preds = []
        for input in tqdm(dataloader):
            input = input.cuda()
            cost = model.model.cost_output(input.x, input.pos, input.mask, input.node_pos, input.node_mask, input.output_idx)
            pred = cost.view(-1)
            preds.append(pred.detach().cpu().numpy())
        preds = np.concatenate(preds)
        preds = dataloader.batch_sampler.group(preds)
        return preds

def best_preds(costs: list[np.ndarray]) -> list[int]:
    return [np.argmax(cost).item() for cost in costs]

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
    query_sampler = BatchedQuerySampler(dataset, args.batch_size)
    dataloader = DataLoader(test_dataset, batch_sampler=query_sampler, collate_fn=model.batch_transformed_samples)
    results = test(model, dataloader)
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
    parser.add_argument('--model', type=str, default='models/cat2_on_imdb_JOB_JOP_val.pth')
    args = parser.parse_args()
    main(args)