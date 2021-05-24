import pickle

from ranker import Ranker
from src.preprocess import utils


# DO NOT MODIFY CLASS NAME
class Searcher:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit. The model 
    # parameter allows you to pass in a precomputed model that is already in 
    # memory for the searcher to use such as LSI, LDA, Word2vec models. 
    # MAKE SURE YOU DON'T LOAD A MODEL INTO MEMORY HERE AS THIS IS RUN AT QUERY TIME.
    def __init__(self, parser, indexer, model=None):
        self._parser = parser
        self._indexer = indexer
        self._ranker = Ranker()
        self._model = model
        self.inverted_index = indexer.inverted_idx

    def relevant_terms_in_inverted_index(self, query):
        """
        this function compare the parsed query to the inverted index
        in order to remove words that not showed in the corpus
        and make fit with the same words that's wrote a little bit differently.
        :param query - parsed query. given in Dictionary.
        :return a Dictionary that's fit the inverted index.
        """
        query_tokens = {}
        for term in query:
            key = term
            if key not in self.inverted_index:
                if key.lower() in self.inverted_index:
                    key = key.lower()
                elif term.upper() in self.inverted_index:
                    key = key.upper()
                else:
                    continue
            query_tokens[key] = query[term]
        return query_tokens

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query, k=2000):
        """ 
        Executes a query over an existing index and returns the number of 
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
        Output:
            A tuple containing the number of relevant search results, and 
            a list of tweet_ids where the first element is the most relavant 
            and the last is the least relevant result.
        """
        query_as_list = self._parser.parse_sentence(query)
        query_as_dic = {}
        for token in query_as_list:
            if token in query_as_dic:
                query_as_dic[token] += 1
            else:
                query_as_dic[token] = 1
        query_as_dic = self.relevant_terms_in_inverted_index(query_as_dic)
        relevant_docs = self._relevant_docs_from_posting(query_as_dic)
        average_doc_length = self.inverted_index['__total_length__'] / self.inverted_index['__doc_number__']
        ranked_doc_ids = Ranker.rank_relevant_docs(relevant_docs, query_as_dic,average_doc_length, k)
        return len(ranked_doc_ids), ranked_doc_ids

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def _relevant_docs_from_posting(self, query):
        """
        This function loads the posting list and count the amount of relevant documents per term.
        :param query_as_list: parsed query tokens
        :return: dictionary of relevant documents mapping doc_id to document frequency.
        """
        inverted_index = self._indexer.inverted_idx  # TODO: get inverted_index.
        relevant_docs = {}
        data = self._indexer.postingDict

        if len(query) == 0:
            return relevant_docs
        for term in query:
            if term not in inverted_index:
                continue
            twit_id = data[term]
            for i in twit_id:
                if i[0] not in relevant_docs:
                    relevant_docs[i[0]] = {term: (i[2], i[3], i[1])}
                else:
                    relevant_docs[i[0]][term] = (i[2], i[3], i[1])

        return relevant_docs
