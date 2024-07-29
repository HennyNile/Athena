import argparse
import sys
import os
import numpy as np
import json
import psycopg2

sys.path.append('.')

from src.utils.dataset_utils import read_dataset, load_Bao_options
from src.Bao.BaoForPostgreSQL.bao_server.caro_utils import prepare_train_dataset, prepare_test_dataset
from src.Bao.BaoForPostgreSQL.bao_server import model
from src.utils.record_utils import load_records, write_records
from src.utils.workload_utils import read_workload
from src.utils.db_utils import DBConn
from src.Bao.generate_dataset import arm_idx_to_hints

def main(args):
    database, workload, plan_method = args.dataset.split('/')
    pretrain_dataset_dir = args.dataset
    eval_workload, eval_dataset_dir = f'{workload}-sample', f'{database}/{workload}-sample/{plan_method}'
    model_dir = f'models/bao_on_{workload}_{plan_method}/'
    model_retrain_dir = f'models/bao_on_{workload}_{plan_method}_retrain/'
    os.makedirs(model_retrain_dir, exist_ok=True)
    
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
    retrain_interval = (int(len(eval_names)/100)+1)*10
    
    with DBConn(database) as db:
        db.prewarm()
        for query_idx, (name, eval_query, x) in enumerate(zip(eval_names, eval_queries, xs)):
            print(f'Evaluate Query {name}')
            record = []
            
            # 1. select plan
            pred = reg.predict(x)
            selected_plan_idx = int(np.argmin(pred))
            
            # 2. check records
            if selected_plan_idx in records[query_idx]:
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
                print('The test result does not exist')
                
                # 3. record doesn't exist, exeucte the query
                selected_option = eval_bao_options[query_idx][selected_plan_idx]
                hints = arm_idx_to_hints(selected_option)
                try:
                    for i in range(test_runtimes):
                        sample = db.get_result(eval_query, hints, timeout=test_timeout)
                        record.append(sample['Execution Time'])
                except psycopg2.errors.QueryCanceled:
                    db.rollback()
                    record = [test_timeout]
                
                # 4. store the records
                records[query_idx][selected_plan_idx] = record
                write_records(database, workload, plan_method, records)
                if len(record) == 5:
                    test_execution_time_list.append(sum(record[2:])/3)
                elif len(record) == 2:
                    test_execution_time_list.append(record[1])
                elif len(record) == 1:
                    test_execution_time_list.append(record[0])
                    
                # 5. update training samples and retrain the model after a certain interval
                train_x.append(eval_dataset[query_idx][selected_plan_idx])
                train_y.append(test_execution_time_list[-1])
                if (query_idx + 1) % retrain_interval == 0:
                    reg.fit(train_x, train_y)
                    model_retrain_epoch_dir = os.path.join(model_retrain_dir, f'epoch_{query_idx+1}/')
                    os.makedirs(model_retrain_epoch_dir, exist_ok=True)
                    reg.save(model_retrain_epoch_dir)
    
    print(f'DB: {database}, Pretrain Workload: {workload}, Eval Workload: {eval_workload},  Plan Generation method: {plan_method}, Model: Bao, test runtime: {sum(test_execution_time_list)}')
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb/JOB/Bao')
    parser.add_argument('--eval', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    main(args)
    