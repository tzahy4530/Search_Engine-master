import math


# you can change whatever you want in this module, just make sure it doesn't
# break the searcher module
class Ranker:
    def __init__(self):
        pass

    @staticmethod
    def rank_relevant_docs(relevant_doc, query_doc,average_doc_length, k):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param relevant_doc: dictionary of documents that contains at least one term from the query.
        :param query_doc: a Dictonary that's contain the frequency of every word in the user query.
        :param k - the document number that the user wanna get back.
        :return: sorted list of documents by score
        """

        relvant_twit_id = relevant_doc.keys()

        square_query_term_weight = 0
        for term in query_doc:
            square_query_term_weight += math.pow(query_doc[term], 2)

        ranked_doc = []

        k = 0.01
        b = 0.80

        for twit_id in relvant_twit_id:
            bm25_rank = 0
            for term in relevant_doc[twit_id]:
                tf = relevant_doc[twit_id][term][0]
                idf = relevant_doc[twit_id][term][1]
                doc_length = relevant_doc[twit_id][term][2]
                bm25_rank += (tf * idf * query_doc[term]*(k+1))/(tf+k*(1-b + (b*doc_length/average_doc_length)))


            if len(ranked_doc) == k:
                if bm25_rank > ranked_doc[k - 1][1]:
                    ranked_doc[k - 1] = (twit_id, bm25_rank)
                    ranked_doc = sorted(ranked_doc, key=lambda x: -x[1])
            else:
                ranked_doc.append((twit_id, bm25_rank))
                ranked_doc = sorted(ranked_doc, key=lambda x: -x[1])

        return [d[0] for d in ranked_doc]

    @staticmethod
    def retrieve_top_k(sorted_relevant_doc, k=1):
        """
        return a list of top K tweets based on their ranking from highest to lowest
        :param sorted_relevant_doc: list of all candidates docs.
        :param k: Number of top document to return
        :return: list of relevant document
        """
        return sorted_relevant_doc
