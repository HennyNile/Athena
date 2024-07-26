#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import json

import psycopg2
from tqdm import tqdm
import math

sys.path.append('.')
from src.utils.db_utils import DBConn
from src.utils.workload_utils import read_workload
from src.utils.plan_utils import UniquePlan, plan_to_leading_hint
from src.utils.record_utils import load_records, write_records
from src.utils.dataset_utils import read_dataset, load_Lero_options, load_Bao_options
from src.utils.result_utils import load_model_selected_plans
from src.utils.result_utils import load_latest_plans
from src.Lero.generate_dataset import SwingOption
from src.Bao.generate_dataset import arm_idx_to_hints


def main(args: argparse.Namespace):
    names, queries = read_workload(args.workload)
    print('load workload finished')
    dataset_path = f'datasets/{args.database}/{args.workload}/{args.plan_generation_method}/'
    names, candidate_plans = read_dataset(dataset_path)
    print('load candidate plans finisehd')
    if args.plan_generation_method == 'Lero':
        names, lero_options = load_Lero_options(args.database, args.workload)
        print('load Lero options finished')
    if args.plan_generation_method == 'Bao':
        names, bao_options = load_Bao_options(args.database, args.workload)
        print('load Bao options finished')
    selected_plans_idx = load_model_selected_plans(args.database, args.workload, args.plan_generation_method, args.result_path)
    print('load model selected plans finished')
    records = load_records(args.database, args.workload, args.plan_generation_method)
    print('load records finished')
    # names, baseline_plans = load_latest_plans(args.database, args.workload, 'pg')
    print('load baseline plans finished')
    test_runtimes = 1
    test_timeout = 1800000
    
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
                print('The test result exists')
                record = records[query_idx][str(selected_plans_idx[query_idx])]
                if len(record) == 5:
                    test_execution_time_list.append(sum(record[2:])/3)
                elif len(record) == 2:
                    test_execution_time_list.append(record[1])
                elif len(record) == 1:
                    test_execution_time_list.append(record[0])
                else:
                    raise ValueError(f'The num of records is not correct, query_idx: {query_idx}, query_name: {name}, records: {record}')
            else:
                print('The test result does not exist')
                
                # set hints based on plan generation method
                if args.plan_generation_method == 'Lero':
                    selected_option = lero_options[query_idx][selected_plans_idx[query_idx]]
                    hints = ["SET enable_lero TO off"]
                    if selected_option is not None:
                        option_obj = SwingOption(selected_option[0], selected_option[1])
                        hints = option_obj.get_hints()
                elif args.plan_generation_method == 'JOP':
                    leading_hint = plan_to_leading_hint(candidate_plans[query_idx][selected_plans_idx[query_idx]])
                    hints = ['SET enable_join_order_plans = off', 'SET geqo_threshold = 20']
                    if not selected_plans_idx == 0:
                        query = f'/*+ Leading({leading_hint}) */ {query}'
                elif args.plan_generation_method == 'Bao':
                    selected_option = bao_options[query_idx][selected_plans_idx[query_idx]]
                    hints = arm_idx_to_hints(selected_option)
                    
                # get test result
                try:
                    for i in range(test_runtimes):
                        sample = db.get_result(query, hints, timeout=test_timeout)
                        record.append(sample['Execution Time'])
                except psycopg2.errors.QueryCanceled:
                    db.rollback()
                    record = [test_timeout]
                records[query_idx][selected_plans_idx[query_idx]] = record
                write_records(args.database, args.workload, args.plan_generation_method, records)
                if len(record) == 5:
                    test_execution_time_list.append(sum(record[2:])/3)
                elif len(record) == 2:
                    test_execution_time_list.append(record[1])
                elif len(record) == 1:
                    test_execution_time_list.append(record[0])
    # compute geometric mean relative latency
    # gmel = [test_execution_time_list[i]/ baseline_plans[i]['Execution Time'] for i in range(len(test_execution_time_list))]
    gmel_value = 1
    # temp = 1
    # for i in range(0, len(gmel)) : 
    #     temp = temp * gmel[i] 
    # temp2 = (float)(math.pow(temp, (1 / len(gmel)))) 
    # gmel_value = (float)(temp2) 
    print(f'DB: {args.database}, Workload: {args.workload}, Plan Generation method: {args.plan_generation_method}, Model: {args.result_path}, test runtime: {sum(test_execution_time_list)}, GMRL: {gmel_value}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='imdb')
    parser.add_argument('--workload', type=str, default='JOB-sample')
    parser.add_argument('--query_begin', type=int, default=0)
    parser.add_argument('--plan_generation_method', type=str, default='JOP')
    parser.add_argument('--model', type=str, default='CAT')
    parser.add_argument('--result_path', type=str, default='CAT')
    args = parser.parse_args()
    main(args)