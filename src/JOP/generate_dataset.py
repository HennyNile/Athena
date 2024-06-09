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
from src.utils.plan_utils import UniquePlan

JOP_plans_filepath = '/tmp/JOP_join_order_plans.txt'

def main(args: argparse.Namespace):
    names, queries = read_workload(args.workload)
    dataset_path = f'datasets/{args.database}/{args.workload}/JOP'
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
            timeout = 0
            samples = []
            print(f"Generate plans of {name}")
            db.get_plan(query, ['SET enable_join_order_plans = on'])
            join_order_hints = list(set(open(JOP_plans_filepath, 'r').readlines()))
            for i in tqdm(range(len(join_order_hints) + 1)):
                hints = ['SET enable_join_order_plans = off']
                if not i == 0:
                    query = f'/*+ Leading({join_order_hints[i-1]}) */ {query}'
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
            print(f"Get {len(samples)} samples.")
            sample_set = []
            for i in range(1, len(samples)):
                plan = UniquePlan(samples[i]['Plan'])
                if plan not in sample_set:
                    sample_set.append(plan)
                else:
                    print(f"Duplicate plan found in {name} query {query_idx}.")
            with open(query_path, 'w') as f:
                json.dump(samples, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='imdb')
    parser.add_argument('--workload', type=str, default='JOB')
    parser.add_argument('--query_begin', type=int, default=0)
    args = parser.parse_args()
    main(args)