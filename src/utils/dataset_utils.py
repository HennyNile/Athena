import os
import json

def timeout_time(default: float) -> float:
    timeout = 4 * default
    if timeout >= 240000:
        timeout = max(default, 240000)
    elif timeout <= 5000:
        timeout = 5000
    return timeout

def read_dataset(dataset_path):
    with open(os.path.join(dataset_path, 'names.json'), 'r') as f:
        names = json.load(f)
    samples = []
    files = os.listdir(dataset_path)
    files = [f for f in files if f != 'names.json']
    files.sort()
    for file in files:
        with open(os.path.join(dataset_path, file), 'r') as f:
            sample = json.load(f)
            for plan in sample:
                if 'Execution Time' not in plan:
                    plan['Timeout Time'] = timeout_time(sample[0]['Execution Time'])
            samples.append(sample)
    return names, samples