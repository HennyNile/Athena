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

from cat import Cat

sys.path.append('.')
from src.utils.dataset_utils import read_dataset
from src.utils.db_utils import DBConn
from src.utils.sampler_utils import QuerySampler, ItemwiseSampler, PairwiseSampler, BalancedPairwiseSampler

class PlanDataset(Dataset):
    def __init__(self, dataset: list[list[dict]], model: Cat):
        plans = [plan for query in dataset for plan in query]
        self.model = model
        self.samples = model.transform(plans)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class CostDataset(Dataset):
    def __init__(self, plan_dataset: PlanDataset, tmp_dir: str):
        if os.path.exists(tmp_dir):
            print(f'Deleting {tmp_dir}')
            shutil.rmtree(tmp_dir)
        print(f'Creating {tmp_dir}')
        os.makedirs(tmp_dir, exist_ok=True)
        model = plan_dataset.model
        model.model.cpu()
        model.model.eval()
        self.tmp_dir = tmp_dir
        self.num_samples = len(plan_dataset.samples)
        print('Preparing cost dataset')
        for idx, sample in enumerate(tqdm(plan_dataset.samples)):
            x, pos, mask, cards, cost = model.transform_sample(sample, torch.device('cpu'))
            obj = {
                'x': x,
                'pos': pos,
                'mask': mask,
                'cards': cards,
                'cost': cost
            }
            torch.save(obj, os.path.join(tmp_dir, f'{idx}.pt'))

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        obj = torch.load(os.path.join(self.tmp_dir, f'{idx}.pt'))
        return obj['x'], obj['pos'], obj['mask'], obj['cards'], obj['cost']

def train_cards(model, optimizer, dataloader, val_dataloader, num_epochs, checkpoint_path, epoch_start = 0):
    for epoch in range(epoch_start, num_epochs):
        model.model.cuda()
        model.model.train()
        losses = []
        for x, pos, mask, cards_label, _ in tqdm(dataloader):
            cards = model.model.cards_output(x, pos, mask)
            cards_mask = mask[:, 0].isfinite() & cards_label.isfinite()
            cards_output_vec = cards[cards_mask]
            cards_label_vec = cards_label[cards_mask]
            weight = cards_label_vec / torch.sum(cards_label_vec)
            weight = weight.clamp(min=1e-3)
            loss = torch.sum(weight * (cards_output_vec - cards_label_vec) ** 2)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = sum(losses) / len(losses)
        writer.add_scalar('train/cards_loss', loss, epoch)
        print(f'Epoch {epoch}, loss: {loss}')

        # model.model.cpu()
        model.model.eval()
        # losses = []
        # qerrors = []
        # for x, pos, mask, cards_label, _ in tqdm(val_dataloader):
        #     cards = model.model.cards_output(x, pos, mask)
        #     cards_mask = mask[:, 0].isfinite() & cards_label.isfinite()
        #     cards_output_vec = cards[cards_mask]
        #     cards_label_vec = cards_label[cards_mask]
        #     weight = cards_label_vec / torch.sum(cards_label_vec)
        #     weight = weight.clamp(min=1e-3)
        #     loss = torch.torch.sum(weight * (cards_output_vec - cards_label_vec) ** 2)
        #     losses.append(loss.item())
        #     qerror = model.q_error_loss(cards_output_vec, cards_label_vec)
        #     qerrors.append(qerror.detach().cpu().numpy())
        # loss = sum(losses) / len(losses)
        # writer.add_scalar('val/cards_loss', loss, epoch)
        # print(f'Validation loss: {sum(losses) / len(losses)}')
        # qerrors = np.concatenate(qerrors)
        # p50 = np.percentile(qerrors, 50)
        # p90 = np.percentile(qerrors, 90)
        # p95 = np.percentile(qerrors, 95)
        # p99 = np.percentile(qerrors, 99)
        # writer.add_scalar('val/p50', np.log10(p50), epoch)
        # writer.add_scalar('val/p90', np.log10(p90), epoch)
        # writer.add_scalar('val/p95', np.log10(p95), epoch)
        # writer.add_scalar('val/p99', np.log10(p99), epoch)
        # print(f'p50: {p50}, p90: {p90}, p95: {p95}, p99: {p99}', flush=True)

def train_cost(model, optimizer, dataloader, val_dataloader, num_epochs, checkpoint_path, epoch_start = 0):
    for epoch in range(epoch_start, num_epochs):
        model.model.cuda()
        model.model.train()
        losses = []
        for x, pos, mask, _, cost_label in tqdm(dataloader):
            x, pos, mask, cost_label = x.cuda(), pos.cuda(), mask.cuda(), cost_label.cuda()
            cost = model.model.cost_output(x, pos, mask)
            pred = cost.view(-1, 2)
            label = cost_label.view(-1, 2)
            pred = pred[:,0] - pred[:,1]
            # label = (label[:,0] > label[:,1]).float()
            label = F.sigmoid((label[:,0].log() - label[:,1].log()) * 2)
            loss = F.binary_cross_entropy_with_logits(pred, label)
            # label_log_diff = label[:,0].log() - label[:,1].log()
            # weight = torch.abs(label_log_diff)
            # weight = weight.clamp(max=math.log(8.))
            # weight = weight / torch.sum(weight)
            # label = F.sigmoid(label_log_diff * 2)
            # loss = F.binary_cross_entropy_with_logits(pred, label, weight, reduction='sum')
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = sum(losses) / len(losses)
        writer.add_scalar('train/cost_loss', loss, epoch)
        print(f'Epoch {epoch}, loss: {loss}')

        model.model.cpu()
        model.model.eval()
        losses = []
        pred_costs = []
        min_costs = []
        for x, pos, mask, _, cost_label in tqdm(val_dataloader):
            cost = model.model.cost_output(x, pos, mask)
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
        table_map, column_map, normalizer = db.get_db_info()

    # create model
    model = Cat(table_map, column_map, normalizer)
    model.fit_all(train_queries + val_queries)
    model.fit_train(train_queries)
    model.init_model()

    train_dataset = PlanDataset(train_queries, model)
    itemwise_sampler = ItemwiseSampler(train_queries, args.batch_size)
    pairwise_sampler = PairwiseSampler(train_queries, args.batch_size // 2)
    val_dataset = PlanDataset(val_queries, model)
    val_sampler = QuerySampler(val_queries)

    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=model.get_collate_fn(torch.device('cuda')), shuffle=True)
    dataloader = DataLoader(train_dataset, batch_sampler=itemwise_sampler, collate_fn=model.get_collate_fn(torch.device('cuda')))
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=model.get_collate_fn(torch.device('cuda')))
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    if args.from_cards is None:
        train_cards(model, optimizer, dataloader, val_dataloader, args.cards_epoch, 'models')
        model.save('cards.pth')
    else:
        model.load(args.from_cards)

    train_dataset = CostDataset(train_dataset, 'tmp_train')
    val_dataset = CostDataset(val_dataset, 'tmp_val')
    dataloader = DataLoader(train_dataset, batch_sampler=pairwise_sampler, collate_fn=model.get_collate_fn2(torch.device('cpu')))
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=model.get_collate_fn2(torch.device('cpu')))
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    train_cost(model, optimizer, dataloader, val_dataloader, args.cost_epoch, 'models')
    model.save('cost.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb/JOB/Bao')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cards_epoch', type=int, default=50)
    parser.add_argument('--cost_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--from_cards', type=str, default=None)
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