import importlib

import pandas as pd
import metrics
import numpy as np
import os
from src import search_engine_best
import re


def invalid_tweet_id(tid):
    tid_ptrn = re.compile('\d+')
    if not isinstance(tid, str):
        tid = str(tid)
    if tid_ptrn.fullmatch(tid) is None:
        return True
    return False


if __name__ == '__main__':
    engine_module = search_engine_best
    se = importlib.import_module(engine_module)
    print(f"Successfully imported module {engine_module}.")
    engine = se.SearchEngine()

    queries = pd.read_csv(os.path.join('data', 'queries_train.tsv'), sep='\t')
    bench_data_path = os.path.join('data', 'benchmark_data_train.snappy.parquet')
    bench_lbls_path = os.path.join('data', 'benchmark_lbls_train.csv')
    queries_path = os.path.join('data', 'queries_train.tsv')

    engine.load_precomputed_model()
    engine.build_index_from_parquet(bench_data_path)

    # model_dir = os.path.join('.', 'model')
    bench_lbls = pd.read_csv(bench_lbls_path,
                             dtype={'query': int, 'tweet': str, 'y_true': int})

    q2n_relevant = bench_lbls.groupby('query')['y_true'].sum().to_dict()
    # queries_results = []
    results=pd.DataFrame({'query_num':[],'map':[],'recall':[],'presision@5':[],'presision@10':[],'presision@50':[],'presision':[]})
    for i, row in queries.iterrows():
        queries_results=[]
        q_id = row['query_id']
        q_keywords = row['keywords']
        q_n_res, q_res = engine.search(q_keywords)
        if q_n_res is None or q_res is None or q_n_res < 1 or len(q_res) < 1:
            print(f"Query {q_id} with keywords '{q_keywords}' returned no results.")
        else:
            print(
                f"search_engine successfully returned {q_n_res} results for query number {q_id}.")
            invalid_tweet_ids = [doc_id for doc_id in q_res if invalid_tweet_id(doc_id)]
            if len(invalid_tweet_ids) > 0:
                print(f"Query  {q_id} returned results that are not valid tweet ids: " + str(
                    invalid_tweet_ids[:10]))
            queries_results.extend(
                [(q_id, str(doc_id)) for doc_id in q_res if not invalid_tweet_id(doc_id)])
        queries_results = pd.DataFrame(queries_results, columns=['query', 'tweet'])

        # merge query results with labels benchmark
        q_results_labeled = None
        if bench_lbls is not None and len(queries_results) > 0:
            q_results_labeled = pd.merge(queries_results, bench_lbls,
                                         on=['query', 'tweet'], how='inner', suffixes=('_result', '_bench'))
            # q_results_labeled.rename(columns={'y_true': 'label'})
            zero_recall_qs = [q_id for q_id, rel in q2n_relevant.items() \
                              if metrics.recall_single(q_results_labeled, rel, q_id) == 0]
            # if len(zero_recall_qs) > 0:
            #     logging.warning(
            #         f"{engine_module}'s recall for the following queries was zero {zero_recall_qs}.")

        if q_results_labeled is not None:
            # test that MAP > 0
            results_map = metrics.map(q_results_labeled)
            # logging.debug(f"{engine_module} results have MAP value of {results_map}.")
            # if results_map <= 0 or results_map > 1:
            #     logging.error(f'{engine_module} results MAP value is out of range (0,1).')

            # test that the average across queries of precision,
            # precision@5, precision@10, precision@50, and recall
            # is in [0,1].
            prec, p5, p10, p50, recall = \
                metrics.precision(q_results_labeled), \
                metrics.precision(q_results_labeled.groupby('query').head(5)), \
                metrics.precision(q_results_labeled.groupby('query').head(10)), \
                metrics.precision(q_results_labeled.groupby('query').head(50)), \
                metrics.recall(q_results_labeled, {i+1:q2n_relevant.get(i+1)})
            print(results_map,prec,p5,p10,p50,recall)
            results=results.append({'query_num':i+1,'map':results_map,'recall':recall,'presision@5':p5,'presision@10':p10,'presision@50':p50,'presision':prec},ignore_index=True)
        results.to_csv(engine_module+'.csv', index=False)
        print(results)