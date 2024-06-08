import sys
import argparse
import os
import json

sys.path.append('.')

from src.utils.db_utils import DBConn
from src.utils.workload_utils import read_workload
from src.utils.time_utils import get_current_time

plan_output_dirpath_template = 'results/{}/{}/pg/{}/'

def main(args: argparse.Namespace):
    db_name, workload = args.database, args.workload
    names, queries = read_workload(args.workload)
    timestamp = get_current_time()
    plan_output_dirpath = plan_output_dirpath_template.format(db_name, workload, timestamp)
    if not os.path.exists(plan_output_dirpath):
        os.makedirs(plan_output_dirpath)
    with DBConn(db_name) as db:
        db.prewarm()
        for name, query in zip(names, queries):
            print(f"Query: {name}")
            result = db.get_result(query)
            result = db.get_result(query)
            plan_output_filepath = plan_output_dirpath + f'/{name}.json'
            json.dump(result, open(plan_output_filepath, 'w'))
            print('Running Time: {}'.format(result['Execution Time']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='imdb')
    parser.add_argument('--workload', type=str, default='JOB')
    args = parser.parse_args()
    main(args)