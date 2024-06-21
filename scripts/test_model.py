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
from src.utils.plan_utils import UniquePlan, plan_to_leading_hint
from src.utils.record_utils import load_records, write_records
from src.utils.dataset_utils import read_dataset
from src.utils.result_utils import load_model_selected_plans


def main(args: argparse.Namespace):
    names, queries = read_workload(args.workload)
    print('load workload finished')
    dataset_path = f'datasets/{args.database}/{args.workload}/{args.plan_generation_method}/'
    names, candidate_plans = read_dataset(dataset_path)
    print('load candidate plans finisehd')
    selected_plans_idx = load_model_selected_plans(args.database, args.workload, args.plan_generation_method, args.model)
    print('load model selected plans finished')
    records = load_records(args.database, args.workload, args.plan_generation_method)
    print('load records finished')
    test_runtimes = 5
    test_timeout = 0
    
    test_execution_time_list = []
    with DBConn(args.database) as db:
        db.prewarm()
        for query_idx, (name, query) in enumerate(zip(names, queries)):
            record = []
            if query_idx < args.query_begin:
                continue
            print(f"Generate plans of {name}")
            assert len(records) >=  query_idx
            if len(records) == query_idx: # append a empty dict for new query
                records.append({})
            if str(selected_plans_idx[query_idx]) in records[query_idx]: # the test result exists
                test_execution_time_list.append(sum(records[query_idx][str(selected_plans_idx[query_idx])][2:])/3)
            else: 
                leading_hint = plan_to_leading_hint(candidate_plans[query_idx][selected_plans_idx[query_idx]])
                query = f'/*+ Leading({leading_hint}) */ {query}'
                hints = ['SET enable_join_order_plans = off', 'SET geqo_threshold = 20']
                for i in range(test_runtimes):
                    sample = db.get_result(query, hints, timeout=test_timeout)
                    record.append(sample['Execution Time'])
                records[query_idx][selected_plans_idx[query_idx]] = record
                write_records(args.database, args.workload, args.plan_generation_method, records)
                test_execution_time_list.append(sum(record[2:])/3)
    print(f'DB: {args.database}, Workload: {args.workload}, Plan Generation method: {args.plan_generation_method}, Model: {args.model}, test runtime: {sum(test_execution_time_list)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='imdb')
    parser.add_argument('--workload', type=str, default='JOB-sample')
    parser.add_argument('--query_begin', type=int, default=0)
    parser.add_argument('--plan_generation_method', type=str, default='JOP')
    parser.add_argument('--model', type=str, default='CAT')
    args = parser.parse_args()
    main(args)