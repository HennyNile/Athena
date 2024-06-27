import sys
import os
import json
from datetime import datetime

sys.path.append('.')

from src.utils.db_utils import DBConn
    
def load_db_column_info(db):
    db_column_info_filepath = f'src/utils/workload_gen/db_column_infos/{db}.json'
    if os.path.exists(db_column_info_filepath):
        with open(db_column_info_filepath, 'r') as f:
            db_column_info = json.load(f)
        return db_column_info
    else:
        return {}

def write_db_column_info(db, db_column_info):
    db_column_info_filepath = f'src/utils/workload_gen/db_column_infos/{db}.json'
    os.makedirs('src/utils/workload_gen/db_column_infos/', exist_ok=True)
    with open(db_column_info_filepath, 'w') as f:
        json.dump(db_column_info, f)

if __name__ == '__main__':
    db = 'stats'
    table = 'posts'
    column = 'CreationDate'
    # with DBConn(db) as dbconn:
    #     print(dbconn.get_column_normalization(table, column))
    d = datetime.strptime('2010-07-21 15:23:53', '%Y-%m-%d %H:%M:%S')
    d2 = datetime.strptime('2010-08-21 15:23:53', '%Y-%m-%d %H:%M:%S')
    print(d.year, d.month, d.day, d.hour, d.minute, d.second)
    delta = d2 - d
    print(delta.days)
    print(delta.seconds)

    print(d.strftime('%Y-%m-%d %H:%M:%S'))