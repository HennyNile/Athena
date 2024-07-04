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

from cat import Lero

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

def min_max_pooling(x: torch.Tensor, indices_batch: list[torch.Tensor]):
    batch_size, _, dim = x.shape
    neg_inf = torch.full((batch_size, 1, dim), -torch.inf, dtype=torch.float32, device=x.device)
    pos_inf = torch.full((batch_size, 1, dim), torch.inf, dtype=torch.float32, device=x.device)
    for i, indices in enumerate(indices_batch):
        use_min = i % 2 == 0
        default = pos_inf if use_min else neg_inf
        pool_fn = torch.min if use_min else torch.max
        x = torch.concat((default, x), dim=1)
        vec_pairs = torch.gather(x, 1, indices[:, :, None].repeat(1, 1, dim)).view(batch_size, -1, 2, dim)
        x = pool_fn(vec_pairs[:, :, 0], vec_pairs[:, :, 1])
    return x.view(batch_size, dim)

def train(model: Lero, optimizer, dataloader, val_dataloader, test_dataloader, num_epochs):
    model.model.cuda()
    for epoch in range(num_epochs):
        model.model.train()
        losses = []
        for trees, cost_label, cards_label, exprs, indices, conds, filters, nodes in tqdm(dataloader):
            exprs = model.model.expr_encoder(exprs)
            exprs = min_max_pooling(exprs, indices)
            batch_size, _, _ = trees[0].shape
            _, expr_dim = exprs.shape
            exprs = torch.concat((torch.zeros(1, expr_dim, dtype=torch.float32, device=exprs.device), exprs), dim=0)
            conds = exprs[conds].view(batch_size, -1, expr_dim)
            filters = exprs[filters].view(batch_size, -1, expr_dim)
            trees = (torch.concat((trees[0], conds.permute(0, 2, 1), filters.permute(0, 2, 1)), dim=1), trees[1])
            cost, cards = model.model(trees, nodes)
            cards_mask = nodes[:,0,1:] & cards_label.isfinite()
            cards_pred_vec = cards[cards_mask]
            cards_label_vec = cards_label[cards_mask]
            weights = (cards_label_vec + model.log_min_true_card)
            weights = weights / weights.sum()
            weights = weights.clamp(min=1e-3)
            cards_loss = torch.sum(weights * (cards_label_vec - cards_pred_vec) ** 2)
            pred = cost.view(-1, 2)
            label = cost_label.view(-1, 2)
            loss = ((label / label.sum(dim=1, keepdim=True)).nan_to_num(1.) * pred.softmax(dim=1)).sum()
            loss = loss + 100 * cards_loss
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
        for trees, cost_label, cards_label, exprs, indices, conds, filters, nodes in tqdm(val_dataloader):
            exprs = model.model.expr_encoder(exprs)
            exprs = min_max_pooling(exprs, indices)
            batch_size, _, _ = trees[0].shape
            _, expr_dim = exprs.shape
            exprs = torch.concat((torch.zeros(1, expr_dim, dtype=torch.float32, device=exprs.device), exprs), dim=0)
            conds = exprs[conds].view(batch_size, -1, expr_dim)
            filters = exprs[filters].view(batch_size, -1, expr_dim)
            trees = (torch.concat((trees[0], conds.permute(0, 2, 1), filters.permute(0, 2, 1)), dim=1), trees[1])
            cost, _ = model.model(trees, nodes)
            # cards_mask = nodes[:,0,1:].isfinite() & cards.isfinite()
            # cards_pred_vec = cards[cards_mask]
            # cards_label_vec = cards_label[cards_mask]
            # weights = (cards_label_vec + model.log_min_true_card)
            # weights = weights / weights.sum()
            # weights = weights.clamp(min=1e-3)
            # cards_loss = torch.sum(weights * (cards_label_vec - cards_pred_vec) ** 2)
            pred = cost.view(-1)
            argmin_pred = torch.argmax(pred)
            pred_cost = cost_label[argmin_pred]
            min_cost = torch.min(cost_label)
            pred_costs.append(pred_cost.item())
            min_costs.append(min_cost.item())
        total_pred_cost = sum(pred_costs)
        total_min_cost = sum(min_costs)
        ability = total_min_cost / total_pred_cost
        writer.add_scalar('val/ability', ability, epoch)
        print(f'Validation ability: {ability * 100}%', flush=True)

        if test_dataloader is not None:
            losses = []
            pred_costs = []
            min_costs = []
            num_timeout = 0
            for trees, cost_label, cards_label, exprs, indices, conds, filters, nodes in tqdm(test_dataloader):
                exprs = model.model.expr_encoder(exprs)
                exprs = min_max_pooling(exprs, indices)
                batch_size, _, _ = trees[0].shape
                _, expr_dim = exprs.shape
                exprs = torch.concat((torch.zeros(1, expr_dim, dtype=torch.float32, device=exprs.device), exprs), dim=0)
                conds = exprs[conds].view(batch_size, -1, expr_dim)
                filters = exprs[filters].view(batch_size, -1, expr_dim)
                trees = (torch.concat((trees[0], conds.permute(0, 2, 1), filters.permute(0, 2, 1)), dim=1), trees[1])
                cost, _ = model.model(trees, nodes)
                # cards_mask = nodes[:,0,1:].isfinite() & cards.isfinite()
                # cards_pred_vec = cards[cards_mask]
                # cards_label_vec = cards_label[cards_mask]
                # weights = (cards_label_vec + model.log_min_true_card)
                # weights = weights / weights.sum()
                # weights = weights.clamp(min=1e-3)
                # cards_loss = torch.sum(weights * (cards_label_vec - cards_pred_vec) ** 2)
                pred = cost.view(-1)
                argmin_pred = torch.argmax(pred)
                pred_cost = cost_label[argmin_pred]
                min_cost = torch.min(cost_label)
                if pred_cost.item() == float('inf'):
                    num_timeout += 1
                    timeout_cost = cost_label[0].item()
                    if timeout_cost >= 240000:
                        timeout_cost = max(cost_label[0], 240000)
                    elif timeout_cost <= 5000:
                        timeout_cost = 5000
                    pred_costs.append(timeout_cost)
                else:
                    pred_costs.append(pred_cost.item())
                min_costs.append(min_cost.item())
            with open('cat3.json', 'w') as f:
                json.dump(pred_costs, f)
            total_pred_cost = sum(pred_costs)
            total_min_cost = sum(min_costs)
            ability = total_min_cost / total_pred_cost
            writer.add_scalar('test/ability', ability, epoch)
            print(f'Test ability: {ability * 100}%, pred time: {total_pred_cost / 1000}, min time: {total_min_cost / 1000}, num timeout: {num_timeout}', flush=True)

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
    model = Lero(db_info)
    model.fit_all(dataset)
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
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.set_printoptions(threshold=10000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    main(args)