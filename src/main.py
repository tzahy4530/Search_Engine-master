from src import search_engine_best
import configuration
import time
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import pandas as pd


if __name__ == '__main__':
    p=pd.read_parquet(r'data/benchmark_data_train.snappy.parquet')
    text=lambda id:p[p['tweet_id']==id].full_text.values
    search_engine = search_engine_best()
    queries=['fauci paper hydroxychloroquine sars','flu kills more than covid','gates implant microchips','Herd immunity reached','children immune to coronavirus']
    search_engine.build_index_from_parquet(r'data/benchmark_data_train.snappy.parquet')
    print ("----------------------")
    print (search_engine.__class__())
    search_engine.load_precomputed_model()
    df=pd.DataFrame({'query_num':[],'twit_id':[],'full_text':[]})
    for query in range(len(queries)):
        number, results= search_engine.search(queries[query])
        results=results[:5]
        for id in results:
            print(id+" : "+text(id))
            df=df.append({'query_num':query+1,'twit_id':id,'full_text':text(id)},ignore_index=True)
    df.to_csv(str(search_engine.__class__)[8:-2] + '_queies.csv', index=False)


