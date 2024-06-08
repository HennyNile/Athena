from functools import cmp_to_key
import os
import re

job_name_re = re.compile(r"(\d+)([a-z]).sql")
number_re = re.compile(r"(\d+).sql")

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

def read_workload(workload: str) -> tuple[list[str], list[str]]:
    path = os.path.join('workloads', workload)
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.sql')]
    key = None
    if workload == 'JOB':
        key = cmp_to_key(job_name_cmp)
    elif workload == 'STATS' or workload == 'TPCH':
        key = cmp_to_key(num_cmp)
    else:
        raise ValueError(f'Invalid workload: {workload}')
    files.sort(key=key)
    names = []
    queries = []
    for file in files:
        names.append(file[:-4])
        with open(os.path.join(path, file)) as f:
            query = f.read()
            queries.append(query)
    return names, queries