#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from cat import Cat

sys.path.append('.')
from src.utils.dataset_utils import read_dataset
from src.utils.db_utils import DBConn

class PlanDataset(Dataset):
    def __init__(self, dataset: list[list[dict]], model: Cat):
        plans = [plan for query in dataset for plan in query]
        self.model = model
        self.samples = model.transform(plans)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
class QuerySampler:
    def __init__(self, dataset: list[list[dict]]):
        num_queries = len(dataset)
        self.all_indices = []
        sample_idx = 0
        for query in dataset:
            self.all_indices.append([sample_idx + i for i in range(len(query))])
            sample_idx += len(query)
        self.current_qid_idx = None

    def __len__(self):
        return len(self.all_indices)

    def __iter__(self):
        self.current_qid_idx = 0
        return self

    def __next__(self):
        if self.current_qid_idx >= len(self.all_indices):
            raise StopIteration
        ret = self.all_indices[self.current_qid_idx]
        self.current_qid_idx += 1
        return ret

class PairwiseSampler:
    def __init__(self, dataset: list[list[dict]], batch_size: int, shuffle = True, drop_last = False):
        self.batch_size = batch_size
        self.all_indices = []
        sample_idx = 0
        for query in dataset:
            for i in range(len(query) - 1):
                for j in range(i + 1, len(query)):
                    if 'Execution Time' in query[i] or 'Execution Time' in query[j]:
                        self.all_indices.append((sample_idx + i, sample_idx + j))
            sample_idx += len(query)
        if drop_last:
            self.num_batches = (len(self.all_indices) + batch_size - 1) // batch_size
        else:
            self.num_batches = len(self.all_indices) // batch_size
        self.shuffled_indices = None
        self.current_batch = None

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.shuffled_indices = self.all_indices[:]
        random.shuffle(self.shuffled_indices)
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        ret = self.shuffled_indices[self.current_batch * self.batch_size: (self.current_batch + 1) * self.batch_size]
        self.current_batch += 1
        return [idx for pair in ret for idx in pair]

def train_cards(model, optimizer, dataloader: DataLoader, val_dataloader, num_epochs, checkpoint_path, epoch_start = 0):
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

        model.model.cpu()
        model.model.eval()
        losses = []
        qerrors = []
        for x, pos, mask, cards_label, _ in tqdm(val_dataloader):
            cards = model.model.cards_output(x, pos, mask)
            cards_mask = mask[:, 0].isfinite() & cards_label.isfinite()
            cards_output_vec = cards[cards_mask]
            cards_label_vec = cards_label[cards_mask]
            weight = cards_label_vec / torch.sum(cards_label_vec)
            weight = weight.clamp(min=1e-3)
            loss = torch.torch.sum(weight * (cards_output_vec - cards_label_vec) ** 2)
            losses.append(loss.item())
            qerror = model.q_error_loss(cards_output_vec, cards_label_vec)
            qerrors.append(qerror.detach().cpu().numpy())
        loss = sum(losses) / len(losses)
        writer.add_scalar('val/cards_loss', loss, epoch)
        print(f'Validation loss: {sum(losses) / len(losses)}')
        qerrors = np.concatenate(qerrors)
        p50 = np.percentile(qerrors, 50)
        p90 = np.percentile(qerrors, 90)
        p95 = np.percentile(qerrors, 95)
        p99 = np.percentile(qerrors, 99)
        writer.add_scalar('val/p50', np.log10(p50), epoch)
        writer.add_scalar('val/p90', np.log10(p90), epoch)
        writer.add_scalar('val/p95', np.log10(p95), epoch)
        writer.add_scalar('val/p99', np.log10(p99), epoch)
        print(f'p50: {p50}, p90: {p90}, p95: {p95}, p99: {p99}', flush=True)

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
    pairwise_sampler = PairwiseSampler(train_queries, args.batch_size // 2)
    val_dataset = PlanDataset(val_queries, model)
    val_sampler = QuerySampler(val_queries)

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=model.get_collate_fn(torch.device('cuda')), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=model.get_collate_fn(torch.device('cpu')))
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-6)
    train_cards(model, optimizer, dataloader, val_dataloader, 500, 'models')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb/JOB/Bao')
    parser.add_argument('--seed', type=int, default=0)
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