#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import os

def main(args):
    model_dirnames = {
        'bao': 'Bao',
        'lero': 'Lero',
        'lero+': 'LeroExp',
        'ours': 'LeroMamba'
    }
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
    methods_dirnames = {
        'bao': 'Bao',
        'lero': 'Lero',
        'ours': 'JOP'
    }
    model_params = {
        "ours": {
            "batch_sizes": {
                'JOB': 512,
                'STATS': 256,
                'TPCH': 256,
                'TPCDS': 256
            },
            "learning_rates": {
                'JOB': 1e-5,
                'STATS': 5e-6,
                'TPCH': 5e-6,
                'TPCDS': 5e-6,
            },
            "epochs": {
                'JOB': 200,
                'STATS': 80,
                'TPCH': 80,
                'TPCDS': 80
            },
            "alpha": {
                'JOB': 1,
                'STATS': 1,
                'TPCH': 0.25,
                'TPCDS': 0.25
            },
            "gamma": {
                'JOB': 0.9,
                'STATS': 0.9,
                'TPCH': 0.05,
                'TPCDS': 0.05,
            }
        }
    }
    for model, workload, method, seed in itertools.product(args.models, args.workloads, args.methods, args.seeds):
        model_dirname = model_dirnames[model]
        dataset_dirname = dataset_dirnames[workload]
        workload_dirname = workload_dirnames[workload]
        method_dirname = methods_dirnames[method]
        batch_sizes = model_params[model]["batch_sizes"]
        learning_rates = model_params[model]["learning_rates"]
        epochs = model_params[model]["epochs"]
        alphas = model_params[model]["alpha"]
        gammas = model_params[model]["gamma"]
        batch_size = batch_sizes[workload]
        learning_rate = learning_rates[workload]
        epoch = epochs[workload]
        alpha = alphas[workload]
        gamma = gammas[workload]
        command = f'./src/{model_dirname}/train.py --dataset {dataset_dirname}/{workload}/{method_dirname} --test {dataset_dirname}/{workload_dirname}/{method_dirname} --valset=empty.json --seed {seed} --epoch {epoch} --batch_size {batch_size} --lr={learning_rate} --logdir=expriments/{model}/{workload}/{method}/{seed} --alpha {alpha} --gamma {gamma}'
        print(command)
        if not args.dry_run:
            os.system(command)

def check_validation(lists, valid_set):
    for item in lists:
        if item not in valid_set:
            return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=lambda s: s.split(','), default=['ours'])
    parser.add_argument('--workloads', type=lambda s: s.split(','), default=['JOB'])
    parser.add_argument('--methods', type=lambda s: s.split(','), default=['ours'])
    parser.add_argument('--seeds', type=lambda s: s.split(','), default=['3407'])
    parser.add_argument('--dry-run', action='store_true', default=False)
    args = parser.parse_args()
    valid_models = {'bao', 'lero', 'lero+', 'ours'}
    valid_workloads = {'JOB', 'STATS', 'TPCH', 'TPCDS'}
    valid_methods = {'bao', 'lero', 'ours'}
    if not check_validation(args.models, valid_models):
        parser.error(f"Invalid model. Choose from {valid_models}")
    if not check_validation(args.workloads, valid_workloads):
        parser.error(f"Invalid workload. Choose from {valid_workloads}")
    if not check_validation(args.methods, valid_methods):
        parser.error(f"Invalid method. Choose from {valid_methods}")
    main(args)