#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse

sys.path.append('.')
from src.utils.dataset_utils import read_dataset

def main(args: argparse.Namespace):
    dataset_path = f'datasets/{args.database}/{args.workload}/{args.method}'
    _, datasets = read_dataset(dataset_path)
    defualt_times = [query[0].get('Execution Time', float('inf')) for query in datasets]
    times = [min([p.get('Execution Time', float('inf')) for p in query]) for query in datasets]
    print(f'Total default min time: {sum(defualt_times)}')
    print(f'Max default min time: {max(defualt_times)}')
    print(f'Total min time: {sum(times)}')
    print(f'Max min time: {max(times)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='imdb')
    parser.add_argument('--workload', type=str, default='JOB')
    parser.add_argument('--method', type=str, default='Bao')
    args = parser.parse_args()
    main(args)