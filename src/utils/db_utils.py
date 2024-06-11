#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psycopg2

class DBConn:
    def __init__(self, db_name):
        self.conn_str = f"dbname={db_name} host=10.16.70.166 user=dbgroup password=fucking_pg"
        self.conn = None

    def __enter__(self):
        self.conn = psycopg2.connect(self.conn_str)
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.conn.close()

    def reset(self):
        cur = self.conn.cursor()
        cur.execute('RESET ALL')

    def rollback(self):
        cur = self.conn.cursor()
        cur.execute('rollback')

    def get_db_info(self):
        cur = self.conn.cursor()
        cur.execute("select table_name from information_schema.tables where table_schema='public'")
        table_map = {}
        column_map = {}
        normalizer = {}
        table_names = [table_name[0] for table_name in cur.fetchall()]
        for table_name in table_names:
            table_map[table_name] = len(table_map)
            cur.execute(f"select column_name, data_type from information_schema.columns where table_schema='public' and table_name = '{table_name}'")
            results = cur.fetchall()
            column_names = [column_name[0] for column_name in results]
            data_types = [data_type[1] for data_type in results]
            for column_name, data_type in zip(column_names, data_types):
                column_map[(table_name, column_name)] = len(column_map)
                if data_type == 'integer':
                    m, M = self.get_column_normalization(table_name, column_name)
                    normalizer[(table_name, column_name)] = (m, M)
        return table_map, column_map, normalizer

    def get_column_normalization(self, table, column):
        cur = self.conn.cursor()
        cur.execute(f"select min({column}), max({column}) from {table}")
        return cur.fetchone()

    def prewarm(self):
        cur = self.conn.cursor()
        cur.execute("select table_name from information_schema.tables where table_schema='public'")
        table_names = [table_name[0] for table_name in cur.fetchall()]
        cur.execute("select indexname from pg_indexes where schemaname='public'")
        index_names = [index_name[0] for index_name in cur.fetchall()]
        for table_name in table_names:
            cur.execute(f"SELECT pg_prewarm('{table_name}')")
        for index_name in index_names:
            cur.execute(f"SELECT pg_prewarm('{index_name}')")

    def get_plan(self, sql, hints = ()):
        cur = self.conn.cursor()
        for hint in hints:
            cur.execute(hint)
        cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
        return cur.fetchone()[0][0]
    
    def get_result(self, sql, hints = (), timeout=None):
        cur = self.conn.cursor()
        for hint in hints:
            cur.execute(hint)
        if timeout is not None:
            cur.execute(f"SET statement_timeout = {timeout}")
        else:
            cur.execute("SET statement_timeout = 0")
        cur.execute(f"EXPLAIN (ANALYZE ON, TIMING OFF, SUMMARY ON, FORMAT JSON) {sql}")
        return cur.fetchone()[0][0]