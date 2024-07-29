import argparse
import sys
import os
import numpy as np
import json

sys.path.append('.')

from src.utils.dataset_utils import read_dataset
from src.Bao.BaoForPostgreSQL.bao_server.caro_utils import prepare_train_dataset, prepare_test_dataset
from src.Bao.BaoForPostgreSQL.bao_server import model

def main(args):
    _, dataset = read_dataset(args.dataset)
    db, workload, plan_method = args.dataset.split('/')
    model_dir = f'models/bao_on_{workload}_{plan_method}/'
        
    if not args.eval:
        x, y = prepare_train_dataset(dataset)
        reg = model.BaoRegression(have_cache_data=False, verbose=True)
        reg.fit(x, y)
        reg.save(model_dir)
    else:
        xs = prepare_test_dataset(dataset)
        print(len(xs))
        reg = model.BaoRegression(have_cache_data=False, verbose=True)
        reg.load(model_dir)
        selected_plan_idxes = []
        for x in xs:
            pred = reg.predict(x)
            selected_plan_idxes.append(int(np.argmin(pred)))
        result_dir = f'results/{db}/{workload}/{plan_method}/'
        os.makedirs(result_dir, exist_ok=True)
        with open(f'{result_dir}Bao.json', 'w+') as f:
            json.dump(selected_plan_idxes, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb/JOB/Bao')
    parser.add_argument('--eval', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    main(args)
    