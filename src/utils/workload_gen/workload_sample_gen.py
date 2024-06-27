import os
import sys
import argparse
import random
from datetime import timedelta, datetime
import psycopg2

sys.path.append('.')

from src.utils.workload_utils import read_workload
from src.utils.db_info_utils import load_db_column_info, write_db_column_info
from src.utils.db_utils import DBConn
from src.utils.result_utils import load_latest_plans

stats_comparison_ops = ['>=', '<=', '=']

def random_date(start, end):
    start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    delta = end - start
    init_delta =  delta.days*24*3600 + delta.seconds
    new_delta = random.randint(0, init_delta)
    new_date = start + timedelta(seconds=new_delta)
    return new_date

def main(args):
    db_column_info = load_db_column_info(args.database)
    names, queries = read_workload(args.workload)
    names, plans = load_latest_plans(args.database, args.workload, 'pg')
    new_queries = [[] for i in range(len(queries))]
    os.makedirs(f'src/utils/workload_gen/samples/{args.workload}-sample/', exist_ok=True)
    with DBConn(args.database) as dbconn:
        dbconn.prewarm()
        for i in range(args.query_begin, len(names)):
            timeout = max(int(plans[i]['Execution Time']), 600000)
            print(timeout)
            trial_num = 0
            new_query_num = 0
            if i == args.query_begin:
                new_query_num = args.start_new_query_num
            query = queries[i]
            tables = [[p.strip() for p in table.split('as')] for table in query.split('FROM')[1].split('WHERE')[0].split(',')]
            tables_alias_2_name = {table[1]: table[0] for table in tables}
            predicates = [pred.strip('; ') for pred in query.split('WHERE')[1].split('AND')]
            while new_query_num < args.sample_num_per_query:
                print(f'Query {i}: {names[i]}, trail num: {trial_num+1}, new query num: {new_query_num}')
                trial_num += 1
                new_query = 'SELECT COUNT(*) FROM' + query.split('FROM')[1].split('WHERE')[0] + 'WHERE' + query.split('WHERE')[1] + ';'
                predicate_stats = {}
                for j in range(len(predicates)):
                    init_pred = predicates[j]
                    if len(init_pred.split('.')) != 2:
                        continue
                    for op in stats_comparison_ops:
                        if op in init_pred:
                            table_alias, column = init_pred.split(op)[0].strip().split('.')
                            table_name = tables_alias_2_name[table_alias]
                            if table_name not in db_column_info:
                                db_column_info[table_name] = {}
                                write_db_column_info(args.database, db_column_info)
                            
                            # obtain max, min value for column
                            if column not in db_column_info[table_name]:
                                min_value, max_value = dbconn.get_column_normalization(table_name, column)
                                if '::timestamp' in init_pred:
                                    min_value = min_value.strftime('%Y-%m-%d %H:%M:%S')
                                    max_value = max_value.strftime('%Y-%m-%d %H:%M:%S')
                                db_column_info[table_name][column] = [min_value, max_value]
                                write_db_column_info(args.database, db_column_info)
                            
                            # check if the column has been used in previous predicates
                            min_value, max_value = db_column_info[table_name][column][0], db_column_info[table_name][column][1]
                            if op == '>=':
                                if table_name in predicate_stats:
                                    if column in predicate_stats[table_name]:
                                        if '<=' in predicate_stats[table_name][column]:
                                            max_value = predicate_stats[table_name][column]['<=']
                            elif op == '<=':
                                if table_name in predicate_stats:
                                    if column in predicate_stats[table_name]:
                                        if '>=' in predicate_stats[table_name][column]:
                                            min_value = predicate_stats[table_name][column]['>=']

                            # generate new predicate
                            if '::timestamp' in init_pred:
                                new_pred_value = random_date(min_value, max_value).strftime('%Y-%m-%d %H:%M:%S')
                                new_pred = f'{table_alias}.{column}{op}\'{new_pred_value}\'::timestamp'
                            else:
                                new_pred_value = random.randint(min_value, max_value)
                                new_pred = f'{table_alias}.{column}{op}{new_pred_value}'
                            

                            # update predicate_stats
                            if table_name not in predicate_stats:
                                predicate_stats[table_name] = {}
                            if column not in predicate_stats[table_name]:
                                predicate_stats[table_name][column] = {}
                            predicate_stats[table_name][column][op] = new_pred_value

                            # replace the old predicate with the new one
                            new_query = new_query.replace(init_pred, new_pred)
                            break # only match one comparison operator
                
                try:
                    row_num = dbconn.get_query_result(new_query, timeout=timeout)[0][0]
                    if row_num > 0:
                        # write new query to file
                        new_queries[i].append(new_query)
                        with open(f'src/utils/workload_gen/samples/{args.workload}-sample/{i:03d}_{new_query_num:02d}.sql', 'w') as f:
                            f.write(new_query)
                        new_query_num += 1
                except Exception as e:
                    dbconn.rollback()
                    continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='imdb')
    parser.add_argument('--workload', type=str, default='JOB')
    parser.add_argument('--query_begin', type=int, default=0)
    parser.add_argument('--sample_num_per_query', type=int, default=1)
    parser.add_argument('--start_new_query_num', type=int, default=0)
    args = parser.parse_args()
    main(args)