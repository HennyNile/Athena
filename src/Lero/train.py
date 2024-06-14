#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from lero import Lero

sys.path.append('.')
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

def train(model, optimizer, dataloader, val_dataloader, num_epochs, checkpoint_path, epoch_start = 0):
    for epoch in range(epoch_start, num_epochs):
        model.model.cuda()
        model.model.train()
        losses = []
        for trees, cost_label in tqdm(dataloader):
            cost = model.model(trees)
            pred = cost.view(-1, 2)
            label = cost_label.view(-1, 2)
            pred = pred[:,0] - pred[:,1]
            label = (label[:,0] > label[:,1]).float()
            loss = F.binary_cross_entropy_with_logits(pred, label)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = sum(losses) / len(losses)
        writer.add_scalar('train/cost_loss', loss, epoch)
        print(f'Epoch {epoch}, loss: {loss}')

        model.model.eval()
        losses = []
        pred_costs = []
        min_costs = []
        for trees, cost_label in tqdm(val_dataloader):
            cost = model.model(trees)
            pred = cost.view(-1)
            argmin_pred = torch.argmin(pred)
            pred_cost = cost_label[argmin_pred]
            min_cost = torch.min(cost_label)
            pred_costs.append(pred_cost.item())
            min_costs.append(min_cost.item())
        total_pred_cost = sum(pred_costs)
        total_min_cost = sum(min_costs)
        ability = total_min_cost / total_pred_cost
        writer.add_scalar('val/ability', ability, epoch)
        print(f'Validation ability: {ability * 100}%', flush=True)

def main(args: argparse.Namespace):
    dataset_regex = re.compile(r'([a-zA-Z0-9]+)/([a-zA-Z0-9]+)/([a-zA-Z0-9]+)')
    database, workload, _ = dataset_regex.match(args.dataset).groups()

    # read dataset
    dataset_path = os.path.join('datasets', args.dataset)
    names, dataset = read_dataset(dataset_path)

    val_names_path = os.path.join('datasets', database, workload, 'val.json')
    with open(val_names_path, 'r') as f:
        val_names = json.load(f)

    # split dataset
    train_queries = [query for name, query in zip(names, dataset) if name not in val_names]
    val_queries = [query for name, query in zip(names, dataset) if name in val_names]

    # read db info
    with DBConn(database) as db:
        table_map, _, _ = db.get_db_info()

    # create model
    model = Lero(table_map)
    model.fit_train(train_queries)
    model.init_model()

    train_dataset = PlanDataset(train_queries, model)
    pairwise_sampler = PairwiseSampler(train_queries, args.batch_size)
    val_dataset = PlanDataset(val_queries, model)
    val_sampler = QuerySampler(val_queries)

    dataloader = DataLoader(train_dataset, batch_sampler=pairwise_sampler, collate_fn=model._transform_samples)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=model._transform_samples)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    train(model, optimizer, dataloader, val_dataloader, args.epoch, 'models')
    model.save('lero.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb/JOB/Bao')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    main(args)