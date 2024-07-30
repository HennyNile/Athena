import argparse
import sys
import os
import numpy as np
import json
import psycopg2
import torch
import random
import copy
from torch.utils.data import DataLoader

sys.path.append('.')

from src.utils.dataset_utils import read_dataset, load_Lero_options, timeout_time
from src.utils.record_utils import load_records, write_records
from src.utils.workload_utils import read_workload
from src.utils.db_utils import DBConn
from src.utils.result_utils import load_latest_plans
from lero import Lero
from src.Lero.train import PlanDataset
from src.Lero.generate_dataset import SwingOption
from src.utils.sampler_utils import QuerySampler, PairwiseSampler
from train import train as lero_train
    
def main(args):
    database, workload, plan_method = args.dataset.split('/')
    pretrain_dataset_dir = os.path.join('datasets', args.dataset)
    eval_workload, eval_dataset_dir = f'{workload}-sample', f'datasets/{database}/{workload}-sample/{plan_method}'
    model_dir = f'models/lero_on_{database}_{workload}_{plan_method}.pth'
    model_retrain_dir = f'models/lero_on_{workload}_{plan_method}_retrain/'
    os.makedirs(model_retrain_dir, exist_ok=True)
    results_path = f'results/{database}/{workload}-sample/{plan_method}/Lero-retrain.json'
    os.makedirs(f'{args.logdir}models/', exist_ok=True)
    
    _, eval_queries = read_workload(eval_workload)
    print('load evaluation workload finished')
    _, pretrain_dataset = read_dataset(pretrain_dataset_dir)
    print('load pretrain dataset finished')
    eval_names, eval_dataset = read_dataset(eval_dataset_dir)
    print('load evaluation dataset finished')
    _, eval_lero_options = load_Lero_options(database, eval_workload)
    print('load Lero options finished')
    records = load_records(database, eval_workload, plan_method)
    print('load records finished')
    with DBConn(database) as db:
        db_info = db.get_db_info()
    model = Lero(db_info.table_map)
    model.init_model()
    model.load(model_dir)
    model.model.cuda()
    print('load pretrained model finished')
    _, baseline_plans = load_latest_plans(database, eval_workload, 'pg')
    print('load baseline plans finished')
    test_execution_time_list = []
    test_runtimes = 1
    test_timeout = 1800000
    explore_timeout = 600000
    retrain_interval = 200
    explore_plan_num = 5
    results = []
    
    assert len(baseline_plans) == len(eval_queries)
    assert len(eval_dataset) == len(eval_queries)
    assert len(eval_lero_options) == len(eval_queries)
    assert len(records) == len(eval_queries)
    
    with DBConn(database) as db:
        db.prewarm()
        train_dataset = copy.deepcopy(pretrain_dataset)
        for query_idx, (name, eval_query, query_dataset) in enumerate(zip(eval_names, eval_queries, eval_dataset)):
            record = []
            new_plans = []
            
            # 0. set exploration timeout
            explore_timeout = timeout_time(baseline_plans[query_idx]['Execution Time'])
            print(f'Evaluate Query {name}, test timeout: {test_timeout}, explore timeout: {explore_timeout}')
            
            # 1. select plan
            model.model.eval()
            query_plandataset = PlanDataset([query_dataset], model)
            query_sampler = QuerySampler([query_dataset])
            query_dataloader = DataLoader(query_plandataset, batch_sampler=query_sampler, collate_fn=model._transform_samples)
            preds = []
            for trees, cost_label, weights in query_dataloader:
                cost = model.model(trees)
                preds.append(cost.view(-1).detach().cpu().numpy())
            assert len(preds) == 1
            pred = preds[0]
            selected_plan_idx = int(np.argmin(pred))
            sorted_plans_idxes = sorted(range(len(pred)), key=lambda i: pred[i])
            explore_plans_idxes = sorted_plans_idxes[1:explore_plan_num]
            results.append(selected_plan_idx)
            
            # 2. check records
            if str(selected_plan_idx) in records[query_idx] and (records[query_idx] != [600000] and records[query_idx] != [-1]):
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
                selected_option = eval_lero_options[query_idx][selected_plan_idx]
                hints = ['SET enable_lero TO off']
                if selected_option is not None:
                    option_obj = SwingOption(selected_option[0], selected_option[1])
                    hints = option_obj.get_hints()
                try:
                    for _ in range(test_runtimes):
                        sample = db.get_result(eval_query, hints, timeout=test_timeout)
                        record.append(sample['Execution Time'])
                except psycopg2.errors.QueryCanceled:
                    db.rollback()
                    record = [test_timeout]
                
                # 4. store the records
                records[query_idx][str(selected_plan_idx)] = record
                write_records(database, eval_workload, plan_method, records)
                if len(record) == 5:
                    test_execution_time_list.append(sum(record[2:])/3)
                elif len(record) == 2:
                    test_execution_time_list.append(record[1])
                elif len(record) == 1:
                    test_execution_time_list.append(record[0])
            # update training data
            new_plan = copy.deepcopy(query_dataset[selected_plan_idx])
            if test_execution_time_list[-1] > explore_timeout:
                new_plan['Timeout Time'] = explore_timeout
            else:
                new_plan['Execution Time'] = test_execution_time_list[-1]
            new_plans.append(new_plan)
            
            # 5. explore new plans
            for explore_plan_idx in explore_plans_idxes:
                explore_record = []
                new_plan, runtime = copy.deepcopy(query_dataset[explore_plan_idx]), 0
                if str(explore_plan_idx) not in records[query_idx]:
                    print(f'The explore result does not exist, exisiting records: {records[query_idx]}, explore plan idx: {explore_plan_idx}')
                    selected_option = eval_lero_options[query_idx][explore_plan_idx]
                    hints = ['SET enable_lero TO off']
                    if selected_option is not None:
                        option_obj = SwingOption(selected_option[0], selected_option[1])
                        hints = option_obj.get_hints()
                    try:
                        for _ in range(test_runtimes):
                            sample = db.get_result(eval_query, hints, timeout=explore_timeout)
                            explore_record.append(sample['Execution Time'])
                        # update training data
                        if len(explore_record) == 5:
                            runtime = sum(explore_record[2:])/3
                        elif len(explore_record) == 2:
                            runtime = explore_record[1]
                        elif len(explore_record) == 1:
                            runtime = explore_record[0]
                    except psycopg2.errors.QueryCanceled:
                        db.rollback()
                        explore_record = [-1]
                        runtime = explore_timeout
                    records[query_idx][str(explore_plan_idx)] = explore_record
                    write_records(database, eval_workload, plan_method, records)
                else:
                    print(f'The explore result exists, exisiting records: {records[query_idx]}, explore plan idx: {explore_plan_idx}')
                    explore_record = records[query_idx][str(explore_plan_idx)]
                    if len(explore_record) == 5:
                        runtime = sum(explore_record[2:])/3
                    elif len(explore_record) == 2:
                        runtime = explore_record[1]
                    elif len(explore_record) == 1 and explore_record[0] != explore_timeout and explore_record[0] != 600000 and explore_record[0] != 1800000 and explore_record[0] != -1:
                        runtime = explore_record[0]
                if runtime > explore_timeout:
                    new_plan['Timeout Time'] = explore_timeout
                else:
                    new_plan['Execution Time'] = runtime
                new_plans.append(new_plan)

            # 6. update training data
            train_dataset.append(new_plans)
   
            # 7. retrain the model after a certain interval
            if (query_idx + 1) % retrain_interval == 0:
                train_plandataset = PlanDataset(train_dataset, model)
                pairwrise_sampler = PairwiseSampler(train_dataset, args.batch_size)
                train_dataloader = DataLoader(train_plandataset, batch_sampler=pairwrise_sampler, collate_fn=model._transform_samples)
                optimizer = torch.optim.Adam(model.model.parameters(), lr=args.lr)
                # initialize a new Lero model
                model.init_model()
                lero_train(model, optimizer, train_dataloader, None, None, args.epoch, args.logdir, 0., 0.)
                model_retrain_epoch_dir = os.path.join(model_retrain_dir, f'epoch_{query_idx+1}/')
                os.makedirs(model_retrain_epoch_dir, exist_ok=True)
                model.save(f'{model_retrain_epoch_dir}/model.pth')

    with open(results_path, 'w') as f:
        json.dump(results, f)
    
    print(f'DB: {database}, Pretrain Workload: {workload}, Eval Workload: {eval_workload},  Plan Generation method: {plan_method}, Model: Lero, test runtime: {sum(test_execution_time_list)}')
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb/JOB/Lero')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--logdir', type=str, default=f'logs/')
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    main(args)
    