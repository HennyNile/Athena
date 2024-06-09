import os
import json

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
            samples.append(sample)
    return names, samples