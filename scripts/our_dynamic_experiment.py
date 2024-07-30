#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys

sys.path.append('.')
from src.LeroMamba.test import test
from src.utils.dataset_utils import read_dataset

def main(args):
    dataset_dirnames = {
        'JOB': 'imdb',
        'STATS': 'stats',
        'TPCH': 'tpch_10',
        'TPCDS': 'tpcds_10'
    }
    workload_dirnames = {
        'JOB': 'JOB-sample',
        'STATS': 'STATS-sample',
        'TPCH': 'TPCH-sample',
        'TPCDS': 'TPCDS-sample'
    }
    batch_sizes = {
        'JOB': 512,
        'STATS': 256,
        'TPCH': 256,
        'TPCDS': 256
    }
    learning_rates = {
        'JOB': 1e-5,
        'STATS': 5e-6,
        'TPCH': 5e-6,
        'TPCDS': 5e-6,
    }
    epochs = {
        'JOB': 200,
        'STATS': 80,
        'TPCH': 80,
        'TPCDS': 80
    }
    alpha = {
        'JOB': 1,
        'STATS': 1,
        'TPCH': 0.25,
        'TPCDS': 0.25
    }
    gamma = {
        'JOB': 0.9,
        'STATS': 0.9,
        'TPCH': 0.05,
        'TPCDS': 0.05,
    }
    dataset = dataset_dirnames[args.workload]
    workload = workload_dirnames[args.workload]
    batch_size = batch_sizes[args.workload]
    learning_rate = learning_rates[args.workload]
    epoch = epochs[args.workload]
    alpha = alpha[args.workload]
    gamma = gamma[args.workload]
    names, train_queries = read_dataset(f'datasets/{dataset}/{args.workload}/JOP')
    base_model = f'expriments/ours/{args.workload}/ours/{args.seed}/models/{epoch - 1}.pt'
    test_names, test_queries = read_dataset(f'datasets/{dataset}/{workload}/JOP', shuffled=True)
    chunk_size = 200
    num_chunks = (len(test_queries) + chunk_size - 1) // chunk_size
    results = []
    for i in range(num_chunks):
        chunk = test_queries[i * chunk_size: (i + 1) * chunk_size]
        names_chunk = test_names[i * chunk_size: (i + 1) * chunk_size]
        topks = test(base_model, dataset, chunk, 5)
        results.extend([topk[0] for topk in topks])

        # update the training dataset
        for q, name, topk in zip(chunk, names_chunk, topks):
            collected_plans = [q[i] for i in topk]
            train_queries.append(collected_plans)
            names.append(name)

        # save the updated training dataset
        update_path = f'datasets/{dataset}/{args.workload}/JOP_update{i + 1}'
        os.makedirs(update_path, exist_ok=True)
        with open(f'{update_path}/names.json', 'w') as f:
            json.dump(names, f)
        for query_idx, q in enumerate(train_queries):
            query_path = os.path.join({update_path}, f'query_{query_idx:04d}.json')
            with open(query_path, 'w') as f:
                json.dump(q, f)
        command = f'./src/LeroMamba/train.py --dataset={dataset}/{args.workload}/JOP --valset=empty.json --seed {args.seed} --epoch {epoch} --batch_size {batch_size} --lr {learning_rate} --logdir=expriments/ours/{args.workload}_update{i + 1}/ours --alpha {alpha} --gamma {gamma}'
        print(command)
        os.system(command)
        base_model = f'expriments/ours/{args.workload}_update{i + 1}/ours/{args.seed}/models/{epoch - 1}.pt'

def check_validation(lists, valid_set):
    for item in lists:
        if item not in valid_set:
            return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='JOB')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    valid_workloads = {'JOB', 'STATS', 'TPCDS'}
    if not check_validation([args.workload], valid_workloads):
        parser.error(f"Invalid workload. Choose from {valid_workloads}")
    main(args)