#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse

sys.path.append('.')
from src.utils.dataset_utils import read_dataset
from src.utils.plan_utils import UniquePlan

def main(args: argparse.Namespace):
    dataset_path1 = f'datasets/{args.database}/{args.workload}/{args.method1}'
    names, datasets1 = read_dataset(dataset_path1)
    dataset_path2 = f'datasets/{args.database}/{args.workload}/{args.method2}'
    _, datasets2 = read_dataset(dataset_path2)
    print([len(query) for query in datasets1])
    default_plans1 = [UniquePlan(query[1]['Plan']) for query in datasets1]
    default_plans2 = [UniquePlan(query[0]['Plan']) for query in datasets2]
    for name, plan1, plan2 in zip(names, default_plans1, default_plans2):
        if plan1 != plan2:
            print(f'{name} is different')
        else:
            print(f'{name} is the same')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='imdb')
    parser.add_argument('--workload', type=str, default='JOB')
    parser.add_argument('--method1', type=str, default='Bao')
    parser.add_argument('--method2', type=str, default='Lero')
    args = parser.parse_args()
    main(args)