#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import re
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from cat2 import Cat, Input

sys.path.append('.')
from src.utils.dataset_utils import read_dataset
from src.utils.db_utils import DBConn
from src.utils.sampler_utils import ItemwiseSampler, BatchedQuerySampler, PairwiseSampler, BalancedPairwiseSampler

class PlanDataset(Dataset):
    def __init__(self, dataset: list[list[dict]], model: Cat, tmp_dir: str = None):
        def get_num_finished(query: list[dict]):
            return sum([1 if 'Execution Time' in plan else 0 for plan in query])
        plans = [plan for query in dataset for plan in query]
        num_finished = [max(1, get_num_finished(query)) for query in dataset for _ in query]
        self.samples = model.transform(plans, num_finished)
        self.model = model
        self.num_samples = len(self.samples)
        self.tmp_dir = tmp_dir
        if tmp_dir is not None:
            if os.path.exists(tmp_dir):
                print(f'Deleting {tmp_dir}')
                shutil.rmtree(tmp_dir)
            print(f'Creating {tmp_dir}')
            os.makedirs(tmp_dir, exist_ok=True)
            print('Preparing plan dataset')
            for idx, sample in enumerate(tqdm(self.samples)):
                input = model.transform_sample(sample)
                input.save(os.path.join(tmp_dir, f'{idx}.pt'))
        else:
            self.inputs = [model.transform_sample(sample) for sample in self.samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.tmp_dir is None:
            # return self.model.transform_sample(self.samples[idx])
            return self.inputs[idx]
        else:
            return Input.load(os.path.join(self.tmp_dir, f'{idx}.pt'))

def train(model, optimizer, dataloader, val_dataloader, test_dataloader, num_epochs, lr_scheduler=None):
    model.model.cuda()
    for epoch in range(num_epochs):
        model.model.train()
        cards_losses = []
        cost_losses = []
        for input in tqdm(dataloader):
            input = input.cuda()
            cost, cards = model.model.cost_and_cards_output(input.x, input.pos, input.mask, input.node_pos, input.node_mask, input.output_idx)
            query_weight = input.weight.unsqueeze(1).repeat(1, input.cards.shape[1])
            cards_mask = input.node_mask[:, 0].isfinite() & input.cards.isfinite()
            cards_output_vec = cards[cards_mask]
            cards_label_vec = input.cards[cards_mask]
            query_weight_vec = query_weight[cards_mask]
            weight = cards_label_vec / torch.sum(cards_label_vec)
            weight = weight.clamp(min=1e-3)
            weight = weight * query_weight_vec
            cards_loss = torch.sum(weight * (cards_output_vec - cards_label_vec) ** 2)
            cards_losses.append(cards_loss.item())
            pred = cost.view(-1, 2)
            label = input.cost.view(-1, 2)
            # pred  = pred[:,0] - pred[:,1]
            # label = (label[:,0] > label[:,1]).float()
            # cost_loss = F.binary_cross_entropy_with_logits(pred, label)
            cost_loss = ((label / label.sum(dim=1, keepdim=True)).nan_to_num(1.) * pred.softmax(dim=1)).sum()
            cost_losses.append(cost_loss.item())
            loss = 1000 * cards_loss + cost_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        cards_loss = sum(cards_losses) / len(cards_losses)
        cost_loss = sum(cost_losses) / len(cost_losses)
        writer.add_scalar('train/cards_loss', cards_loss, epoch)
        writer.add_scalar('train/cost_loss', cost_loss, epoch)
        print(f'Epoch {epoch}, cards_loss: {cards_loss}, cost_loss: {cost_loss}', flush=True)

        model.model.eval()
        cards_losses = []
        qerrors = []
        preds = []
        labels = []
        for input in tqdm(val_dataloader):
            input = input.cuda()
            cost, cards = model.model.cost_and_cards_output(input.x, input.pos, input.mask, input.node_pos, input.node_mask, input.output_idx)
            query_weight = input.weight.unsqueeze(1).repeat(1, input.cards.shape[1])
            cards_mask = input.node_mask[:, 0].isfinite() & input.cards.isfinite()
            cards_output_vec = cards[cards_mask]
            cards_label_vec = input.cards[cards_mask]
            query_weight_vec = query_weight[cards_mask]
            weight = cards_label_vec / torch.sum(cards_label_vec)
            weight = weight.clamp(min=1e-3)
            weight = weight * query_weight_vec
            cards_loss = torch.sum(weight * (cards_output_vec - cards_label_vec) ** 2)
            cards_losses.append(cards_loss.item())
            qerror = model.q_error_loss(cards_output_vec, cards_label_vec)
            qerrors.append(qerror.detach().cpu().numpy())
            pred = cost.view(-1)
            preds.append(pred.detach().cpu().numpy())
            labels.append(input.cost.detach().cpu().numpy())
        if len(cards_losses) != 0:
            cards_loss = sum(cards_losses) / len(cards_losses)
            qerrors = np.concatenate(qerrors)
            p50 = np.percentile(qerrors, 50)
            p90 = np.percentile(qerrors, 90)
            p95 = np.percentile(qerrors, 95)
            p99 = np.percentile(qerrors, 99)
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            preds = val_dataloader.batch_sampler.group(preds)
            labels = val_dataloader.batch_sampler.group(labels)
            argmin_pred = [np.argmax(pred).item() for pred in preds]
            pred_costs = [label[min_pred].item() for label, min_pred in zip(labels, argmin_pred)]
            min_costs = [np.min(label).item() for label in labels]
            total_pred_cost = sum(pred_costs)
            total_min_cost = sum(min_costs)
            ability = total_min_cost / total_pred_cost
            writer.add_scalar('val/cards_loss', cards_loss, epoch)
            writer.add_scalar('val/p50', np.log10(p50), epoch)
            writer.add_scalar('val/p90', np.log10(p90), epoch)
            writer.add_scalar('val/p95', np.log10(p95), epoch)
            writer.add_scalar('val/p99', np.log10(p99), epoch)
            writer.add_scalar('val/ability', ability, epoch)
            print(f'Validation cards_loss: {cards_loss}, p50: {p50}, p90: {p90}, p95: {p95}, p99: {p99}, ability: {ability * 100}%', flush=True)

        if test_dataloader is not None:
            preds = []
            labels = []
            for input in tqdm(test_dataloader):
                input = input.cuda()
                cost = model.model.cost_output(input.x, input.pos, input.mask, input.node_pos, input.node_mask, input.output_idx)
                pred = cost.view(-1)
                preds.append(pred.detach().cpu().numpy())
                labels.append(input.cost.detach().cpu().numpy())
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            preds = test_dataloader.batch_sampler.group(preds)
            labels = test_dataloader.batch_sampler.group(labels)
            argmin_pred = [np.argmax(pred).item() for pred in preds]
            pred_costs = [label[min_pred].item() for label, min_pred in zip(labels, argmin_pred)]
            default_costs = [label[0].item() for label in labels]
            def get_timeout_cost(default):
                timeout = 4 * default
                if timeout >= 240000:
                    return max(default, 240000)
                elif timeout <= 5000:
                    return 5000
                else:
                    return timeout
            pred_costs = [cost if cost != float('inf') else get_timeout_cost(default) for cost, default in zip(pred_costs, default_costs)]
            timeouts = [get_timeout_cost(default) for cost, default in zip(pred_costs, default_costs) if cost == float('inf')]
            min_costs = [np.min(label).item() for label in labels]
            total_pred_cost = sum(pred_costs)
            total_min_cost = sum(min_costs)
            ability = total_min_cost / total_pred_cost
            writer.add_scalar('test/ability', ability, epoch)
            print(f'Test ability: {ability * 100}%, pred time: {total_pred_cost / 1000}, min time: {total_min_cost / 1000}, timeout: {timeouts}', flush=True)

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
    model = Cat(db_info)
    model.fit_all(train_queries + val_queries)
    model.fit_train(train_queries)
    model.init_model()

    train_dataset = PlanDataset(train_queries, model)
    pairwise_sampler = PairwiseSampler(train_queries, args.batch_size // 2)
    val_dataset = PlanDataset(val_queries, model)
    val_sampler = BatchedQuerySampler(val_queries, args.batch_size)

    dataloader = DataLoader(train_dataset, batch_sampler=pairwise_sampler, collate_fn=model.batch_transformed_samples, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=model.batch_transformed_samples, num_workers=8)
    if args.test != '':
        _, test_queries = read_dataset(os.path.join('datasets', args.test))
        test_dataset = PlanDataset(test_queries, model, 'tmp_test')
        test_sampler = BatchedQuerySampler(test_queries, args.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=model.batch_transformed_samples, num_workers=8)
    else:
        test_dataloader = None
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    train(model, optimizer, dataloader, val_dataloader, test_dataloader, args.epoch)
    os.makedirs('models', exist_ok=True)
    model.save(f'models/cat2_on_{database}_{workload}_{method}_{args.valset.split(".")[0]}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb/JOB/Bao')
    parser.add_argument('--test', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--valset', type=str, default='val.json')
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
    torch.set_printoptions(threshold=10_000)
    main(args)