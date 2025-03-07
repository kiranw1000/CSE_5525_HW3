import sqlite3
import numpy as np
import os
import re
import pickle
import random
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any

import torch

DB_PATH = 'data/flight_database.db'

def compute_metrics(gt_path: str, model_path: str, gt_query_records: str = None, model_query_records: str = None):
    '''
    Main function to compute the three metrics used for evaluation: 
        * Exact match for SQL queries
        * Exact match for database records returned by queries
        * F1 score for database records returned by queries

    Inputs:
        * gt_path (str): The path to the ground-truth SQL queries corresponding to the text prompts
        * model_path (str): The path to SQL queries generated by the model, conditioned on the same text prompts
        * gt_query_records (str): If provided, it should be a path to a pickle file containing a list of records
                                  returned by the ground-truth SQL queries.
        * model_query_records (str): If provided, it should be a path to a pickle file containing a list of records
                                     returned by the model-generated SQL queries.
    '''
    gt_qs, gt_records, _ = load_queries_and_records(gt_path, gt_query_records)
    model_qs, model_records, model_error_msgs = load_queries_and_records(model_path, model_query_records)

    sql_em = compute_sql_exact_match(gt_qs, model_qs)
    record_em = compute_record_exact_match(gt_records, model_records)
    record_f1 = compute_record_F1(gt_records, model_records)

    return sql_em, record_em, record_f1, model_error_msgs

def load_queries_and_records(sql_path: str, record_path: str):
    '''
    Helper function for loading saved SQL queries and for computing the
    dataset records associated with said queries.

    Inputs:
        * sql_path (str): Path to a .sql file containing SQL queries
        * record_path (str): If provided, a path to a .pkl file containing dataset
                             records associated with each SQL query in sql_path.
    '''
    read_qs = read_queries(sql_path)

    if record_path is not None:
        with open(record_path, 'rb') as f:
            records, error_msgs = pickle.load(f)
    else:
        records, error_msgs = compute_records(read_qs)

    return read_qs, records, error_msgs

def save_queries_and_records(sql_queries: List[str], sql_path: str, record_path: str):
    '''
    Helper function to save model generated SQL queries and their associated records
    to the specified paths.

    Inputs: 
        * sql_queries (List[str]): The list of SQL queries to save
        * sql_path (str): Path to save SQL queries
        * record_path (str): Path to save database records associated with queries
    '''
    # First save the queries
    with open(sql_path, 'w') as f:
        for query in sql_queries:
            f.write(f'{query}\n')

    # Next compute and save records
    records, error_msgs = compute_records(sql_queries)    
    with open(record_path, 'wb') as f:
        pickle.dump((records, error_msgs), f)

def read_queries(sql_path: str):
    with open(sql_path, 'r') as f:
        qs = [q.strip().split("</s>")[0] for q in f.readlines()]
    return qs

def compute_records(processed_qs: List[str]):
    '''
    Helper function for computing the records associated with each SQL query in the
    input list. You may change the number of threads or the timeout variable (in seconds)
    based on your computational constraints.

    Input:
        * processed_qs (List[str]): The list of SQL queries to execute
    '''
    num_threads = 10
    timeout_secs = 120

    pool = ThreadPoolExecutor(num_threads)
    futures = []
    for i, query in enumerate(processed_qs):
        futures.append(pool.submit(compute_record, i, query))
        
    rec_dict = {}
    try:
        for x in tqdm(as_completed(futures, timeout=timeout_secs)):
            query_id, rec, error_msg = x.result()
            rec_dict[query_id] = (rec, error_msg)
    except:
        for future in futures:
            if not future.done():
                future.cancel()
            
    recs = []
    error_msgs = []
    for i in range(len(processed_qs)):
        if i in rec_dict:
            rec, error_msg = rec_dict[i]
            recs.append(rec)
            error_msgs.append(error_msg)
        else:
            recs.append([])
            error_msgs.append("Query timed out")
            
    return recs, error_msgs

def compute_record(query_id, query):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(query)
        rec = cursor.fetchall()
        error_msg = ""
    except Exception as e:
        rec = []
        error_msg = f"{type(e).__name__}: {e}"

    conn.close()
    return query_id, rec, error_msg

def compute_sql_exact_match(gt_qs: List[str], model_qs: List[str]):
    '''
    Helper function to compute exact match between ground-truth
    and model generated SQL queries.
    '''
    total = 0
    ems = 0
    for gt_q, model_q in zip(gt_qs, model_qs):
        total += 1
        ems += 1 if gt_q == model_q else 0
    return ems / total

def compute_record_exact_match(gt_records: List[Any], model_records: List[Any]):
    '''
    Helper function to compute exact match between records
    generated by ground-truth and model SQL queries
    '''
    total = 0
    ems = 0
    for gt_rec, model_rec in zip(gt_records, model_records):
        total += 1
        ems += 1 if set(gt_rec) == set(model_rec) else 0
    return ems / total

def compute_record_F1(gt_records: List[Any], model_records: List[Any]):
    '''
    Helper function to compute F1 between records
    generated by ground-truth and model SQL queries
    '''
    F1s = []
    for gt_rec, model_rec in zip(gt_records, model_records):
        gt_set = set(gt_rec)
        model_set = set(model_rec)        

        precision_total = len(model_set)
        if precision_total == 0:
            precision = 1
        else:
            precision = len([rec for rec in model_set if rec in gt_set]) / precision_total
    
        recall_total = len(gt_set)    
        if recall_total == 0:
            recall = 1
        else:
            recall = len([rec for rec in gt_set if rec in model_set]) / recall_total

        F1 = 2 * precision * recall / (precision + recall + 1e-8)
        F1s.append(F1)

    return np.mean(F1s)

def set_random_seeds(seed_value=42):
    '''
    Set random seeds for better reproducibility
    '''
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_pickle(pkl_path):
    '''
    Helper function to print the contents of a pickle file
    '''
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data
