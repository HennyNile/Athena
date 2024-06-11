#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json

sys.path.append('.')
import src.Bao.generate_dataset as Bao
from src.utils.db_utils import DBConn
from src.utils.plan_utils import UniquePlan

hints = Bao.arm_idx_to_hints(0)
# hints = []
print(hints)

with open('workloads/JOB/4b.sql', 'r') as f:
    query = f.read()

with DBConn('imdb') as db:
    result = db.get_plan(query, hints)
    plan = UniquePlan(result['Plan'])
    print(plan)

with open('datasets/imdb/JOB/Bao/query_0012.json', 'r') as f:
    plans = json.load(f)
    plan0 = UniquePlan(plans[0]['Plan'])

print(plan0)