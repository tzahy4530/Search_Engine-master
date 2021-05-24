"""
This search engine version using word2vec module that trained on all part A corpus
we used this method to expand the query
An addition to Word2Vec we using stemming
"""
import multiprocessing
import numpy as np
import pandas as pd
from src.preprocess.reader import ReadFile
from src.preprocess.parser_module import Parse
from src.preprocess.indexer import Indexer
from src.retrieval.searcher_word2vec import Searcher
import gensim.models
import warnings
import os
from configuration import ConfigClass


def indexer_multiprocess(indexer, document_list):
    for parsed_document in document_list:
        indexer.add_new_doc(parsed_document)


def func(df):
    parser = df[1]
    df = df[0]
    return df.T.apply(lambda row: parser.parse_doc(row.values))


def parallelize_dataframe(df, parser_object, n_cores=max(multiprocessing.cpu_count() - 3, 1)):
    # n_cores=1
    df_split = np.array_split(df, n_cores)
    pool = multiprocessing.Pool(n_cores)
    df = pd.concat(pool.map(func, zip(df_split, [parser_object for i in df_split])))
    pool.close()
    pool.join()
    return df


class SearchEngine:
    def __init__(self, config=None):
        self._config = config
        if config is not None:
            config.toStem = True;
        self._parser = Parse(config = self._config)
        self._indexer = Indexer(config)
        self._model = None

    def build_index_from_parquet(self, fn):
        """
        Reads parquet file and passes it to the parser, then indexer.
        Input:
            fn - path to parquet file
        Output:
            No output, just modifies the internal _indexer object.
        """
        r = ReadFile().read_file(fn)
        indexer = self._indexer
        p = self._parser
        documents_list = r
        indexer.empty_pendling()
        # parse the document
        document_parsed_list = parallelize_dataframe(documents_list, p).values.tolist()
        for parsed_document in document_parsed_list:
            indexer.add_new_doc(parsed_document)
        indexer.empty_pendling()
        indexer.building_posting_file()
        indexer.save_index("idx_bench.pkl")


    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        self._indexer.load_index(fn)


    def load_precomputed_model(self, model_dir=None):
        """
        Loads a pre-computed model (or models) so we can answer queries.
        This is where you would load models like word2vec, LSI, LDA, etc. and
        assign to self._model, which is passed on to the searcher at query time.
        """
        warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim.models')
        warnings.filterwarnings(action='ignore', category=FutureWarning, module='gensim.models')
        self._model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_dir,'trained_model'), binary=True,
                                                                      encoding='utf-8', unicode_errors='ignore', )

    def search(self, query):
        """
        Executes a query over an existing index and returns the number of
        relevant docs and an ordered list of search results.
        Input:
            query - string.
        Output:
            A tuple containing the number of relevant search results, and
            a list of tweet_ids where the first element is the most relavant
            and the last is the least relevant result.
        """
        searcher = Searcher(self._parser, self._indexer, model=self._model)
        return searcher.search(query)
