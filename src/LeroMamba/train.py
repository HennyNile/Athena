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
from src.utils.time_utils import get_current_time
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

def train(model, optimizer, scheduler, dataloader, val_dataloader, test_dataloader, num_epochs, logdir, alpha, gamma):
    with open(os.path.join(logdir, 'output.txt'), 'a') as logfile:
        for epoch in range(num_epochs):
            model.model.cuda()
            model.model.train()
            losses = []
            for trees, cost_label, weights in tqdm(dataloader):
                cost = model.model(trees.to('cuda'))
                pred = cost.view(-1, 2)
                label = cost_label.view(-1, 2)
                weights = weights.view(-1, 2)
                weights = torch.abs(weights[:,0] - weights[:,1])
                weights = weights ** alpha
                weights = weights / weights.sum()
                pred = pred[:,0] - pred[:,1]
                label = (label[:,0] > label[:,1]).float()
                loss = tailr_loss_with_logits(pred, label, weights, gamma)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.save(os.path.join(logdir, 'models', f'{epoch}.pt'))
            if scheduler is not None:
                scheduler.step()
            loss = sum(losses) / len(losses)
            writer.add_scalar('train/cost_loss', loss, epoch)
            logfile.write(f'Epoch {epoch}, loss: {loss}\n')

            model.model.eval()
            losses = []
            pred_costs = []
            timeout_costs = []
            min_costs = []
            for trees, cost_label, weights in tqdm(val_dataloader):
                cost = model.model(trees.to('cuda'))
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
            if len(pred_costs) != 0:
                total_pred_cost = sum(pred_costs)
                total_min_cost = sum(min_costs)
                ability = total_min_cost / total_pred_cost
                writer.add_scalar('val/ability', ability, epoch)
                logfile.write(f'Validation ability: {ability * 100}%, pred time: {total_pred_cost / 1000}, min time: {total_min_cost / 1000}, {len(timeout_costs)} timeouts: {[c / 1000 for c in timeout_costs]}\n')

            if test_dataloader is not None:
                losses = []
                selected = []
                pred_costs = []
                timeout_costs = []
                min_costs = []
                for trees, cost_label, weights in tqdm(test_dataloader):
                    cost = model.model(trees.to('cuda'))
                    pred = cost.view(-1)
                    argmin_pred = torch.argmin(pred)
                    selected.append(argmin_pred.item())
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
                with open(os.path.join(logdir, 'logs', f'{epoch}.json'), 'w') as f:
                    json.dump(selected, f)
                total_pred_cost = sum(pred_costs)
                total_min_cost = sum(min_costs)
                ability = total_min_cost / total_pred_cost
                writer.add_scalar('test/ability', ability, epoch)
                logfile.write(f'Test ability: {ability * 100}%, pred time: {total_pred_cost / 1000}, min time: {total_min_cost / 1000}, {len(timeout_costs)} timeouts: {[c / 1000 for c in timeout_costs]}\n')
            logfile.flush()

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
    if args.val != '':
        _, val_queries = read_dataset(os.path.join('datasets', args.val))
        val_dataset = PlanDataset(val_queries, model)
        val_sampler = QuerySampler(val_queries)
        val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=model._transform_samples)

    if args.test != '':
        _, test_queries = read_dataset(os.path.join('datasets', args.test))
        test_dataset = PlanDataset(test_queries, model)
        test_sampler = QuerySampler(test_queries)
        test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=model._transform_samples)
    else:
        test_dataloader = None
    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.lr)
    def lr_lambda(warmup=50, decay=30, max_lr=200):
        def ret(epoch):
            if epoch < warmup:
                return math.exp(epoch / warmup * math.log(max_lr))
            else:
                return (math.cos(((epoch - warmup) / decay) * math.pi) + 1) / 2 * (max_lr - 1) + 1
        return ret
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda())
    train(model, optimizer, scheduler, dataloader, val_dataloader, test_dataloader, args.epoch, args.logdir, args.alpha, args.gamma)
    os.makedirs('models', exist_ok=True)
    model.save(f'models/lero_on_{database}_{workload}_{method}_{args.valset.split(".")[0]}.pth')

if __name__ == '__main__':
    timestamp_str = get_current_time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb/JOB/Bao')
    parser.add_argument('--val', type=str, default='')
    parser.add_argument('--test', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--valset', type=str, default='val.json')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--logdir', type=str, default=f'logs/{timestamp_str}')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.9)
    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.join(args.logdir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, 'logs'), exist_ok=True)
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