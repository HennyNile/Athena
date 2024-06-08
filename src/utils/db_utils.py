#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psycopg2
import json

# PG_CONNECTION_STR_JOB = "dbname=imdb host=127.0.0.1"
PG_CONNECTION_STR_JOB = "dbname=imdb host=10.16.70.166 user=dbgroup password=fucking_pg"

_ALL_OPTIONS = [
    "enable_nestloop", "enable_hashjoin", "enable_mergejoin",
    "enable_seqscan", "enable_indexscan", "enable_indexonlyscan"
]

all_48_hint_sets = '''hashjoin,indexonlyscan
hashjoin,indexonlyscan,indexscan
hashjoin,indexonlyscan,indexscan,mergejoin
hashjoin,indexonlyscan,indexscan,mergejoin,nestloop
hashjoin,indexonlyscan,indexscan,mergejoin,seqscan
hashjoin,indexonlyscan,indexscan,nestloop
hashjoin,indexonlyscan,indexscan,nestloop,seqscan
hashjoin,indexonlyscan,indexscan,seqscan
hashjoin,indexonlyscan,mergejoin
hashjoin,indexonlyscan,mergejoin,nestloop
hashjoin,indexonlyscan,mergejoin,nestloop,seqscan
hashjoin,indexonlyscan,mergejoin,seqscan
hashjoin,indexonlyscan,nestloop
hashjoin,indexonlyscan,nestloop,seqscan
hashjoin,indexonlyscan,seqscan
hashjoin,indexscan
hashjoin,indexscan,mergejoin
hashjoin,indexscan,mergejoin,nestloop
hashjoin,indexscan,mergejoin,nestloop,seqscan
hashjoin,indexscan,mergejoin,seqscan
hashjoin,indexscan,nestloop
hashjoin,indexscan,nestloop,seqscan
hashjoin,indexscan,seqscan
hashjoin,mergejoin,nestloop,seqscan
hashjoin,mergejoin,seqscan
hashjoin,nestloop,seqscan
hashjoin,seqscan
indexonlyscan,indexscan,mergejoin
indexonlyscan,indexscan,mergejoin,nestloop
indexonlyscan,indexscan,mergejoin,nestloop,seqscan
indexonlyscan,indexscan,mergejoin,seqscan
indexonlyscan,indexscan,nestloop
indexonlyscan,indexscan,nestloop,seqscan
indexonlyscan,mergejoin
indexonlyscan,mergejoin,nestloop
indexonlyscan,mergejoin,nestloop,seqscan
indexonlyscan,mergejoin,seqscan
indexonlyscan,nestloop
indexonlyscan,nestloop,seqscan
indexscan,mergejoin
indexscan,mergejoin,nestloop
indexscan,mergejoin,nestloop,seqscan
indexscan,mergejoin,seqscan
indexscan,nestloop
indexscan,nestloop,seqscan
mergejoin,nestloop,seqscan
mergejoin,seqscan
nestloop,seqscan'''

all_48_hint_sets = all_48_hint_sets.split('\n')
all_48_hint_sets = [ ["enable_"+j for j in i.split(',')] for i in all_48_hint_sets]

def arm_idx_to_hints(arm_idx):
    hints = []
    for option in _ALL_OPTIONS:
        hints.append(f"SET {option} TO off")

    if arm_idx >= 1 and arm_idx < 49:
        for i in all_48_hint_sets[arm_idx - 1]:
            hints.append(f"SET {i} TO on")
    elif arm_idx == 0:
        for option in _ALL_OPTIONS:
            hints.append(f"SET {option} TO on") # default PG setting 
    else:
        print('48 hint set error')
        exit(0)
    return hints

def prewarm():
    with psycopg2.connect(PG_CONNECTION_STR_JOB) as conn:
        cur = conn.cursor()
        cur.execute("select table_name from information_schema.tables where table_schema='public'")
        table_names = [table_name[0] for table_name in cur.fetchall()]
        cur.execute("select indexname from pg_indexes where schemaname='public'")
        index_names = [index_name[0] for index_name in cur.fetchall()]
        for table_name in table_names:
            cur.execute(f"SELECT pg_prewarm('{table_name}')")
        for index_name in index_names:
            cur.execute(f"SELECT pg_prewarm('{index_name}')")

def hints_to_plan(sql: str, hints: list[str]):
    with psycopg2.connect(PG_CONNECTION_STR_JOB) as conn:
        cur = conn.cursor()
        for hint in hints:
            cur.execute(hint)
        cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
        return cur.fetchone()[0][0]

def hints_to_sample(sql: str, hints: list[str], timeout=None):
    with psycopg2.connect(PG_CONNECTION_STR_JOB) as conn:
        cur = conn.cursor()
        for hint in hints:
            cur.execute(hint)
        if timeout is not None:
            cur.execute(f"SET statement_timeout = {timeout}")
        else:
            cur.execute("SET statement_timeout = 0")
        cur.execute(f"EXPLAIN (ANALYZE ON, TIMING OFF, SUMMARY ON, FORMAT JSON) {sql}")
        return cur.fetchone()[0][0]
    
def run_sql(sql: str, timeout=None):
    with psycopg2.connect(PG_CONNECTION_STR_JOB) as conn:
        cur = conn.cursor()
        if timeout is not None:
            cur.execute(f"SET statement_timeout = {timeout}")
        else:
            cur.execute("SET statement_timeout = 0")
        cur.execute(f"EXPLAIN (ANALYZE ON, TIMING OFF, SUMMARY ON, FORMAT JSON) {sql}")
        return cur.fetchall()[0][0]


def all_plans(query: str) -> tuple[list[dict], list[int]]:
    plans = []
    arms = []
    plan_set = set()
    for arm in range(49):
        hints = arm_idx_to_hints(arm)
        plan = hints_to_plan(query, hints)
        plan_str = json.dumps(plan)
        if plan_str not in plan_set:
            plan_set.add(plan_str)
            plans.append(plan)
            arms.append(arm)
    return plans, arms