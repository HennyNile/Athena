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

def train(model, optimizer, dataloader, val_dataloader, test_dataloader, num_epochs):
    for epoch in range(num_epochs):
        model.model.cuda()
        model.model.train()
        losses = []
        for trees, cost_label in tqdm(dataloader):
            cost = model.model(trees)
            pred = cost.view(-1, 2)
            label = cost_label.view(-1, 2)
            # loss = ((label / label.sum(dim=1, keepdim=True)).nan_to_num(1.) * pred.softmax(dim=1)).sum()
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
        if len(pred_costs) != 0:
            total_pred_cost = sum(pred_costs)
            total_min_cost = sum(min_costs)
            ability = total_min_cost / total_pred_cost
            writer.add_scalar('val/ability', ability, epoch)
            print(f'Validation ability: {ability * 100}%', flush=True)

        if test_dataloader is not None:
            losses = []
            pred_costs = []
            timeout_costs = []
            min_costs = []
            for trees, cost_label in tqdm(test_dataloader):
                cost = model.model(trees)
                pred = cost.view(-1)
                argmin_pred = torch.argmin(pred)
                pred_cost = cost_label[argmin_pred]
                min_cost = torch.min(cost_label)
                if pred_cost.item() == float('inf'):
                    default = cost_label[0].item()
                    timeout_cost = 4 * default
                    if timeout_cost >= 240000:
                        timeout_cost = max(default, 240000)
                    elif timeout_cost <= 5000:
                        timeout_cost = 5000
                    pred_costs.append(timeout_cost)
                    timeout_costs.append(timeout_cost)
                else:
                    pred_costs.append(pred_cost.item())
                min_costs.append(min_cost.item())
            total_pred_cost = sum(pred_costs)
            total_min_cost = sum(min_costs)
            ability = total_min_cost / total_pred_cost
            writer.add_scalar('test/ability', ability, epoch)
            print(f'Test ability: {ability * 100}%, pred time: {total_pred_cost / 1000}, min time: {total_min_cost / 1000}, {len(timeout_costs)} timeouts: {[c / 1000 for c in timeout_costs]}', flush=True)

def main(args: argparse.Namespace):
    dataset_regex = re.compile(r'([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)')
    database, workload, method = dataset_regex.match(args.dataset).groups()

    # read dataset
    dataset_path = os.path.join('datasets', args.dataset)
    names, dataset = read_dataset(dataset_path)

    val_names_path = os.path.join('datasets', database, workload, args.valset)
    with open(val_names_path, 'r') as f:
        val_names = json.load(f)

    # split dataset
    train_queries = [query for name, query in zip(names, dataset) if name not in val_names]
    val_queries = [query for name, query in zip(names, dataset) if name in val_names]

    # read db info
    with DBConn(database) as db:
        db_info = db.get_db_info()

    # create model
    model = Lero(db_info.table_map)
    model.fit_train(train_queries)
    model.init_model()

    train_dataset = PlanDataset(train_queries, model)
    pairwise_sampler = PairwiseSampler(train_queries, args.batch_size)
    val_dataset = PlanDataset(val_queries, model)
    val_sampler = QuerySampler(val_queries)

    dataloader = DataLoader(train_dataset, batch_sampler=pairwise_sampler, collate_fn=model._transform_samples)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=model._transform_samples)
    if args.test != '':
        _, test_queries = read_dataset(os.path.join('datasets', args.test))
        test_dataset = PlanDataset(test_queries, model)
        test_sampler = QuerySampler(test_queries)
        test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=model._transform_samples)
    else:
        test_dataloader = None
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    train(model, optimizer, dataloader, val_dataloader, test_dataloader, args.epoch)
    os.makedirs('models', exist_ok=True)
    model.save(f'models/lero_on_{database}_{workload}_{method}_{args.valset.split(".")[0]}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb/JOB/Bao')
    parser.add_argument('--test', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--valset', type=str, default='val.json')
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(args.cuda)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    main(args)