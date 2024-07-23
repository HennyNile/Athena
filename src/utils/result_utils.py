import os
from functools import cmp_to_key
import sys
import json
import re

job_name_re = re.compile(r"(\d+)([a-z]).json")
number_re = re.compile(r"(\d+).json")
sample_name_re = re.compile(r"(\d+)_(\d+).json")

def job_name_cmp(a: str, b: str) -> int:
    a_id, a_letter = job_name_re.match(a).groups()
    b_id, b_letter = job_name_re.match(b).groups()
    if a_id != b_id:
        return int(a_id) - int(b_id)
    return ord(a_letter) - ord(b_letter)

def num_cmp(a: str, b: str) -> int:
    a_id = number_re.match(a).group(1)
    b_id = number_re.match(b).group(1)
    return int(a_id) - int(b_id)

def sample_name_cmp(a: str, b: str) -> int:
    a_id, a_sample_id = sample_name_re.match(a).groups()
    b_id, b_sample_id = sample_name_re.match(b).groups()
    if a_id != b_id:
        return int(a_id) - int(b_id)
    return int(a_sample_id) - int(b_sample_id)

def load_latest_plans(db, workload, method):
    plan_dirpath = f"results/{db}/{workload}/{method}/"
    timestamp = sorted(os.listdir(plan_dirpath))[-1]
    plan_dirpath = plan_dirpath + timestamp + '/'
    plan_filenames = os.listdir(plan_dirpath)
    key = None
    if workload == 'JOB':
        key = cmp_to_key(job_name_cmp)
    elif workload == 'STATS':
        key = cmp_to_key(num_cmp)
    elif workload == 'JOB-sample' or workload == 'STATS-sample' or workload == 'TPCH' or workload == 'TPCH-sample' or workload == 'TPCDS' or workload == 'TPCDS-sample':
        key = cmp_to_key(sample_name_cmp)
    else:
        raise ValueError(f'Invalid workload: {workload}')
    plan_filenames.sort(key=key)
    names = []
    json_plans = []
    for plan_filename in plan_filenames:
        names.append(plan_filename[:-5])
        with open(os.path.join(plan_dirpath, plan_filename)) as f:
            json_plan = json.loads(f.read())
            json_plans.append(json_plan)
    return names, json_plans

def load_model_selected_plans(db, workload, method, result_path):
    selected_plans_filepath = f'results/{db}/{workload}/{method}/{result_path}.json'
    with open(selected_plans_filepath, 'r') as f:
        selected_plans = json.load(f)
    return selected_plans
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='imdb')
    parser.add_argument('--workload', type=str, default='JOB')
    parser.add_argument('--method', type=str, default='pg')
    args = parser.parse_args()
    names, plans = load_latest_plans(args.database, args.workload, args.method)
    runtime_list = []
    runtime_sum = 0
    for plan in plans:
        runtime_sum += plan['Execution Time']
        runtime_list.append(plan['Execution Time'])
    sorted_runtime_list = sorted(runtime_list)
    print(runtime_sum)
