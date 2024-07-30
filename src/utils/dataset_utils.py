import os
import json

def timeout_time(default: float) -> float:
    timeout = 4 * default
    if timeout >= 240000:
        timeout = max(default, 240000)
    elif timeout <= 5000:
        timeout = 5000
    return timeout

def read_dataset(dataset_path, true_card=False):
    with open(os.path.join(dataset_path, 'names.json'), 'r') as f:
        names = json.load(f)
    samples = []
    files = os.listdir(dataset_path)
    files = [f for f in files if f.startswith('query_')]
    files.sort()
    for file in files:
        with open(os.path.join(dataset_path, file), 'r') as f:
            sample = json.load(f)
            if true_card:
                new_sample = []
                for plan in sample:
                    if 'Execution Time' in plan:
                        new_sample.append(plan)
                samples.append(new_sample)
            elif 'Execution Time' in sample[0]:
                for plan in sample:
                    if 'Execution Time' not in plan:
                        plan['Timeout Time'] = timeout_time(sample[0]['Execution Time'])
                samples.append(sample)
            else:
                samples.append(sample)
    return names, samples

def load_Lero_options(db, workload):
    with open(os.path.join('datasets', db, workload, 'Lero', 'names.json'), 'r') as f:
        names = json.load(f)
    options = []
    files = os.listdir(os.path.join('datasets', db, workload, 'Lero'))
    files = [f for f in files if f.startswith('option_')]
    files.sort()
    for file in files:
        with open(os.path.join('datasets', db, workload, 'Lero', file), 'r') as f:
            option = json.load(f)
            options.append(option)
    return names, options

def load_Bao_options(db, workload):
    with open(os.path.join('datasets', db, workload, 'Bao', 'names.json'), 'r') as f:
        names = json.load(f)
    options = []
    files = os.listdir(os.path.join('datasets', db, workload, 'Bao'))
    files = [f for f in files if f.startswith('option_')]
    files.sort()
    for file in files:
        with open(os.path.join('datasets', db, workload, 'Bao', file), 'r') as f:
            option = json.load(f)
            options.append(option)
    return names, options

def load_training_order(db, workload):
    with open(os.path.join('datasets', db, workload, 'shuffled.json'), 'r') as f:
        training_order = json.load(f)
    return training_order

if __name__ == '__main__':
    names, options = load_Lero_options('imdb', 'JOB-sample')