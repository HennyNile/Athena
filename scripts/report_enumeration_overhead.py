#!/usr/bin/env python3

import argparse
import itertools
import os
import sys

sys.path.append('.')
from src.utils.db_utils import DBConn
from src.utils.workload_utils import read_workload

def main(args):
    methods = ('pg', 'bao', 'lero', 'ours')
    methods = ('pg',)
    workloads = ('JOB', )
    databases = {'JOB': 'imdb', 'STATS': 'stats', 'TPCDS': 'tpcds_10'}
    workload_dirs = {'JOB': 'JOB-sample', 'STATS': 'STATS-sample', 'TPCDS': 'TPCDS-sample'}
    for method, workload in itertools.product(methods, workloads):
        database = databases[workload]
        workload_dir = workload_dirs[workload]
        _, workload_set = read_workload(workload_dir)
        with DBConn(database) as db:
            if method == 'pg':
                hints = []
            elif method == 'bao':
                hints = ['SET enable_bao TO on']
            elif method == 'lero':
                hints = ['SET enable_lero TO on']
            else:
                hints = ['SET enable_join_order_plans = on', 'SET geqo_threshold = 20']
            for query in workload_set:
                plan = db.get_plan(query, hints)
                print(plan['Planning Time'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)