import json
import os

def load_records(db, workload, method) -> list[dict]: 
    records_filepath = f'records/{db}/{workload}/{method}.json'
    if os.path.exists(records_filepath):
        with open(records_filepath, 'r') as f:
            records = json.load(f)
        return records
    else:
        return []

def write_records(db, workload, method, records):
    records_filepath = f'records/{db}/{workload}/{method}.json'
    records_dirpath = f'records/{db}/{workload}/'
    os.makedirs(records_dirpath, exist_ok=True)
    with open(records_filepath, 'w') as f:
        json.dump(records, f)
