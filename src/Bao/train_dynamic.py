import argparse
import sys
import os
import numpy as np
import json
import psycopg2
import torch
import random

sys.path.append('.')
sys.path.append('./src/Bao/BaoForPostgreSQL/bao_server')

from src.utils.dataset_utils import read_dataset, load_Bao_options
from src.Bao.BaoForPostgreSQL.bao_server.caro_utils import prepare_train_dataset, prepare_test_dataset
from src.Bao.BaoForPostgreSQL.bao_server import model
from src.utils.record_utils import load_records, write_records
from src.utils.workload_utils import read_workload
from src.utils.db_utils import DBConn
from src.Bao.generate_dataset import arm_idx_to_hints

def main(args):
    database, workload, plan_method = args.dataset.split('/')
    pretrain_dataset_dir = os.path.join('datasets', args.dataset)
    eval_workload, eval_dataset_dir = f'{workload}-sample', f'datasets/{database}/{workload}-sample/{plan_method}'
    model_dir = f'models/bao_on_{workload}_{plan_method}/'
    model_retrain_dir = f'models/bao_on_{workload}_{plan_method}_retrain/'
    os.makedirs(model_retrain_dir, exist_ok=True)
    results_path = f'results/{database}/{workload}-sample/{plan_method}/Bao-retrain.json'
    
    _, eval_queries = read_workload(eval_workload)
    print('load evaluation workload finished')
    _, pretrain_dataset = read_dataset(pretrain_dataset_dir)
    print('load pretrain dataset finished')
    eval_names, eval_dataset = read_dataset(eval_dataset_dir)
    print('load evaluation dataset finished')
    _, eval_bao_options = load_Bao_options(database, eval_workload)
    print('load Bao options finished')
    records = load_records(database, eval_workload, plan_method)
    print('load records finished')
    reg = model.BaoRegression(have_cache_data=False, verbose=True)
    reg.load(model_dir)
    print('load pretrained model finished')
    train_x, train_y = prepare_train_dataset(pretrain_dataset)
    xs = prepare_test_dataset(eval_dataset)
    test_execution_time_list = []
    test_runtimes = 1
    test_timeout = 1800000
    explore_timeout = 600000
    retrain_interval = (int(len(eval_names)/100)+1)*10
    explore_plan_num = 5
    results = []
    
    with DBConn(database) as db:
        db.prewarm()
        for query_idx, (name, eval_query, x) in enumerate(zip(eval_names, eval_queries, xs)):
            print(f'Evaluate Query {name}')
            record = []
            
            # 1. select plan
            pred = reg.predict(x)
            sorted_plans_idxes = sorted(range(len(pred)), key=lambda i: pred[i])
            explore_plans_idxes = sorted_plans_idxes[1:explore_plan_num]
            selected_plan_idx = int(np.argmin(pred))
            results.append(selected_plan_idx)
            
            # 2. check records
            if str(selected_plan_idx) in records[query_idx]:
                print('The test result exists')
                record = records[query_idx][str(selected_plan_idx)]
                if len(record) == 5:
                    test_execution_time_list.append(sum(record[2:])/3)
                elif len(record) == 2:
                    test_execution_time_list.append(record[1])
                elif len(record) == 1:
                    test_execution_time_list.append(record[0])
                else:
                    raise ValueError(f'The num of records is not correct, query_idx: {query_idx}, query_name: {name}, records: {record}')
            else:
                print(f'The test result does not exist, exisiting records: {records[query_idx]}, selected plan idx: {selected_plan_idx}')
                
                # 3. record doesn't exist, exeucte the query
                selected_option = eval_bao_options[query_idx][selected_plan_idx]
                hints = arm_idx_to_hints(selected_option)
                try:
                    for _ in range(test_runtimes):
                        sample = db.get_result(eval_query, hints, timeout=test_timeout)
                        record.append(sample['Execution Time'])
                except psycopg2.errors.QueryCanceled:
                    db.rollback()
                    record = [test_timeout]
                
                # 4. store the records
                records[query_idx][selected_plan_idx] = record
                write_records(database, eval_workload, plan_method, records)
                if len(record) == 5:
                    test_execution_time_list.append(sum(record[2:])/3)
                elif len(record) == 2:
                    test_execution_time_list.append(record[1])
                elif len(record) == 1:
                    test_execution_time_list.append(record[0])
            # update training data
            train_x.append(eval_dataset[query_idx][selected_plan_idx])
            train_y.append(test_execution_time_list[-1])
            
            # 5. explore new plans
            for explore_plan_idx in explore_plans_idxes:
                explore_record = []
                if str(explore_plan_idx) not in records[query_idx]:
                    print(f'The explore result does not exist, exisiting records: {records[query_idx]}, explore plan idx: {explore_plan_idx}')
                    selected_option = eval_bao_options[query_idx][explore_plan_idx]
                    hints = arm_idx_to_hints(selected_option)
                    try:
                        for _ in range(test_runtimes):
                            sample = db.get_result(eval_query, hints, timeout=explore_timeout)
                            explore_record.append(sample['Execution Time'])
                        # update training data
                        if len(explore_record) == 5:
                            train_x.append(eval_dataset[query_idx][explore_plan_idx])
                            train_y.append(sum(explore_record[2:])/3)
                        elif len(record) == 2:
                            train_x.append(eval_dataset[query_idx][explore_plan_idx])
                            train_y.append(record[1])
                        elif len(record) == 1 and record[0] != explore_timeout:
                            train_x.append(eval_dataset[query_idx][explore_plan_idx])
                            train_y.append(record[0])
                    except psycopg2.errors.QueryCanceled:
                        db.rollback()
                        explore_record = [explore_timeout]
                    records[query_idx][explore_plan_idx] = explore_record
                    write_records(database, eval_workload, plan_method, records)
                else:
                    print(f'The explore result exists, exisiting records: {records[query_idx]}, explore plan idx: {explore_plan_idx}')
                    explore_record = records[query_idx][str(explore_plan_idx)]
                    if len(explore_record) == 5:
                        train_x.append(eval_dataset[query_idx][explore_plan_idx])
                        train_y.append(sum(explore_record[2:])/3)
                    elif len(explore_record) == 2:
                        train_x.append(eval_dataset[query_idx][explore_plan_idx])
                        train_y.append(explore_record[1])
                    elif len(explore_record) == 1 and explore_record[0]:
                        train_x.append(eval_dataset[query_idx][explore_plan_idx])
                        train_y.append(explore_record[0])
                
   
            # 6. retrain the model after a certain interval
            if (query_idx + 1) % retrain_interval == 0:
                reg.fit(train_x, train_y)
                model_retrain_epoch_dir = os.path.join(model_retrain_dir, f'epoch_{query_idx+1}/')
                os.makedirs(model_retrain_epoch_dir, exist_ok=True)
                reg.save(model_retrain_epoch_dir)

    with open(results_path, 'w') as f:
        json.dump(results, f)
    
    print(f'DB: {database}, Pretrain Workload: {workload}, Eval Workload: {eval_workload},  Plan Generation method: {plan_method}, Model: Bao, test runtime: {sum(test_execution_time_list)}')
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb/JOB/Bao')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main(args)
    