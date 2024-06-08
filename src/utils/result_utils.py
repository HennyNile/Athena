import os
from functools import cmp_to_key
import sys
import json
import re

job_name_re = re.compile(r"(\d+)([a-z]).json")
number_re = re.compile(r"(\d+).json")

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


def load_latest_plans(db, workload, method):
    plan_dirpath = f"results/{db.lower()}/{workload.lower()}/{method.lower()}/"
    timestamp = sorted(os.listdir(plan_dirpath))[-1]
    plan_dirpath = plan_dirpath + timestamp + '/'
    plan_filenames = os.listdir(plan_dirpath)
    key = None
    if workload == 'JOB':
        key = cmp_to_key(job_name_cmp)
    elif workload == 'STATS' or workload == 'TPCH':
        key = cmp_to_key(num_cmp)
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='imdb')
    parser.add_argument('--workload', type=str, default='JOB')
    parser.add_argument('--method', type=str, default='pg')
    args = parser.parse_args()
    names, plans = load_latest_plans(args.database, args.workload, args.method)
    print(len(plans))
    print(names)
    print(plans[0])


