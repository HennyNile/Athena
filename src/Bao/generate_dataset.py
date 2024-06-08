#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import json

import psycopg2
from tqdm import tqdm

sys.path.append('.')
from src.utils.db_utils import DBConn
from src.utils.workload_utils import read_workload

ALL_OPTIONS = [
    "enable_nestloop", "enable_hashjoin", "enable_mergejoin",
    "enable_seqscan", "enable_indexscan", "enable_indexonlyscan"
]

all_48_hint_sets = '''hashjoin,indexonlyscan
hashjoin,indexonlyscan,indexscan
hashjoin,indexonlyscan,indexscan,mergejoin
hashjoin,indexonlyscan,indexscan,mergejoin,nestloop
hashjoin,indexonlyscan,indexscan,mergejoin,seqscan
hashjoin,indexonlyscan,indexscan,nestloop
hashjoin,indexonlyscan,indexscan,nestloop,seqscan
hashjoin,indexonlyscan,indexscan,seqscan
hashjoin,indexonlyscan,mergejoin
hashjoin,indexonlyscan,mergejoin,nestloop
hashjoin,indexonlyscan,mergejoin,nestloop,seqscan
hashjoin,indexonlyscan,mergejoin,seqscan
hashjoin,indexonlyscan,nestloop
hashjoin,indexonlyscan,nestloop,seqscan
hashjoin,indexonlyscan,seqscan
hashjoin,indexscan
hashjoin,indexscan,mergejoin
hashjoin,indexscan,mergejoin,nestloop
hashjoin,indexscan,mergejoin,nestloop,seqscan
hashjoin,indexscan,mergejoin,seqscan
hashjoin,indexscan,nestloop
hashjoin,indexscan,nestloop,seqscan
hashjoin,indexscan,seqscan
hashjoin,mergejoin,nestloop,seqscan
hashjoin,mergejoin,seqscan
hashjoin,nestloop,seqscan
hashjoin,seqscan
indexonlyscan,indexscan,mergejoin
indexonlyscan,indexscan,mergejoin,nestloop
indexonlyscan,indexscan,mergejoin,nestloop,seqscan
indexonlyscan,indexscan,mergejoin,seqscan
indexonlyscan,indexscan,nestloop
indexonlyscan,indexscan,nestloop,seqscan
indexonlyscan,mergejoin
indexonlyscan,mergejoin,nestloop
indexonlyscan,mergejoin,nestloop,seqscan
indexonlyscan,mergejoin,seqscan
indexonlyscan,nestloop
indexonlyscan,nestloop,seqscan
indexscan,mergejoin
indexscan,mergejoin,nestloop
indexscan,mergejoin,nestloop,seqscan
indexscan,mergejoin,seqscan
indexscan,nestloop
indexscan,nestloop,seqscan
mergejoin,nestloop,seqscan
mergejoin,seqscan
nestloop,seqscan'''

all_48_hint_sets = all_48_hint_sets.split('\n')
all_48_hint_sets = [ ["enable_"+j for j in i.split(',')] for i in all_48_hint_sets]

def arm_idx_to_hints(arm_idx: int) -> list[str]:
    hints = []
    for option in ALL_OPTIONS:
        hints.append(f"SET {option} TO off")

    if arm_idx >= 1 and arm_idx < 49:
        for i in all_48_hint_sets[arm_idx - 1]:
            hints.append(f"SET {i} TO on")
    elif arm_idx == 0:
        for option in ALL_OPTIONS:
            hints.append(f"SET {option} TO on") # default PG setting 
    else:
        print('48 hint set error')
        exit(0)
    return hints

def main(args: argparse.Namespace):
    names, queries = read_workload(args.workload)
    dataset_path = f'datasets/{args.database}/{args.workload}/Bao'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    names_path = os.path.join(dataset_path, 'names.json')
    with open(names_path, 'w') as f:
        json.dump(names, f)
    with DBConn(args.database) as db:
        db.prewarm()
        for query_idx, (name, query) in enumerate(zip(names, queries)):
            if query_idx < args.query_begin:
                continue
            plan_set = set()
            arms = []
            for arm in range(49):
                hints = arm_idx_to_hints(arm)
                plan = db.get_plan(query, hints)
                plan_str = json.dumps(plan)
                if plan_str not in plan_set:
                    plan_set.add(plan_str)
                    arms.append(arm)
            timeout = 0
            samples = []
            print(f"Generate plans of {name}")
            for arm in tqdm(arms):
                hints = arm_idx_to_hints(arm)
                try:
                    try:
                        db.get_result(query, hints, timeout=timeout)
                    except psycopg2.errors.QueryCanceled:
                        db.rollback()
                    sample = db.get_result(query, hints, timeout=timeout)
                    if timeout == 0:
                        timeout = 4 * sample['Execution Time']
                        if timeout >= 240000:
                            timeout = max(sample['Execution Time'], 240000)
                        elif timeout <= 5000:
                            timeout = 5000
                except psycopg2.errors.QueryCanceled:
                    sample = db.get_plan(query, hints)
                    db.rollback()
                samples.append(sample)
            query_path = os.path.join(dataset_path, f'query_{query_idx:04d}.json')
            with open(query_path, 'w') as f:
                json.dump(samples, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='imdb')
    parser.add_argument('--workload', type=str, default='JOB')
    parser.add_argument('--query_begin', type=int, default=0)
    args = parser.parse_args()
    main(args)