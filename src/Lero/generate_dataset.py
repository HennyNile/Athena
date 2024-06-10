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
from src.utils.plan_utils import UniquePlan, SCAN_TYPES, JOIN_TYPES

SAME_CARD_TYPES = ["Hash", "Materialize", "Sort", "Incremental Sort", "Limit"]
OP_TYPES = ["Aggregate", "Bitmap Index Scan"] +  SCAN_TYPES + JOIN_TYPES + SAME_CARD_TYPES

class SwingOption:
    def __init__(self, num_tables: int, swing_factor: float):
        self.num_tables = num_tables
        self.swing_factor = swing_factor

    def get_hints(self):
        hints = []
        hints.append('SET enable_lero TO on')
        hints.append(f'SET lero_subquery_table_num TO {self.num_tables}')
        hints.append(f'SET lero_swing_factor TO {self.swing_factor}')
        return hints

    def replace(self, plan):
        input_card = None
        input_tables = []
        output_card = None

        if "Plans" in plan:
            children = plan['Plans']
            child_input_tables = None
            if len(children) == 1:
                child_input_card, child_input_tables = self.replace(children[0])
                input_card = child_input_card
                input_tables += child_input_tables
            else:
                for child in children:
                    _, child_input_tables = self.replace(child)
                    input_tables += child_input_tables

        node_type = plan['Node Type']
        if node_type in JOIN_TYPES:
            if len(input_tables) == self.num_tables:
                plan['Plan Rows'] /= self.swing_factor
            output_card = plan['Plan Rows']
        elif node_type in SAME_CARD_TYPES:
            if input_card is not None:
                plan['Plan Rows'] = input_card
                output_card = input_card
        elif node_type in SCAN_TYPES:
            input_tables.append(plan['Relation Name'])
        elif node_type not in self.OP_TYPES:
            raise Exception("Unknown node type " + node_type)

        return output_card, input_tables

def main(args: argparse.Namespace):
    names, queries = read_workload(args.workload)
    dataset_path = f'datasets/{args.database}/{args.workload}/Lero'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    names_path = os.path.join(dataset_path, 'names.json')
    with open(names_path, 'w') as f:
        json.dump(names, f)
    with DBConn(args.database) as db:
        db.prewarm()
        db.reset()
        for query_idx, (name, query) in enumerate(zip(names, queries)):
            if query_idx < args.query_begin:
                continue
            plan_set = set()
            options = []
            default_plan = UniquePlan(db.get_plan(query)['Plan'])
            plan_set.add(default_plan)
            options.append(None)
            for num_tables in range(default_plan.num_tables, 1, -1):
                for swing_factor in (0.1, 100., 10., 0.01):
                    option = SwingOption(num_tables, swing_factor)
                    hints = option.get_hints()
                    plan = UniquePlan(db.get_plan(query, hints)['Plan'])
                    if plan not in plan_set:
                        plan_set.add(plan)
                        options.append(option)
            timeout = 0
            samples = []
            print(f"Generate plans of {name}")
            for option in tqdm(options):
                hints = option.get_hints() if option is not None else ["SET enable_lero TO off"]
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