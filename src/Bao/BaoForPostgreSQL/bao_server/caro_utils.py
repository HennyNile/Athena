import json
import os

def prepare_train_dataset(dataset):
    x, y = [], []
    for sample in dataset:
        for plan in sample:
            if 'Execution Time' in plan:
                x.append(json.dumps(plan))
                y.append(plan['Execution Time'])
    return x, y

def prepare_test_dataset(dataset):
    return dataset