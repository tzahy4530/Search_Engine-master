import math
import pickle


class Indexer:
    """
    Creating inverted index as Dictionary - key= term , value = tuple(number of appear in entire corpus,df,line number in posting file)
    an addition to creating posting files - ordered by first char of the term (in case that the term isn't word it will save in 'simbols.pkl' file
    """

    def __init__(self, config):
        self.inverted_idx = {
            '__doc_number__': 0, '__total_length__': 0}  # key -> tuple(number of appear in entire corpus,df,line number in posting file,file_ID)
        self.config = config
        self.__garage_dict = {0: {}}  # will save word untile arriving to limit then its draing to posting file
        self.__pending_list = {}
        self.__inverted_doc = {}
        self.__doc_counter = 0
        self.__drain_docs_thread = []
        self.postingDict = self.__garage_dict[0]
        self.__document_list = []

    def changeTerm(self, old, new):
        """

        :param old: older term
        :param new: target to change to
        :return:
        """
        if old not in self.inverted_idx or new == old:
            return
        temp = self.inverted_idx.pop(old)
        self.inverted_idx[new] = temp

    def add_new_doc(self, document):

        terms_dic = document.term_doc_dictionary
        key_list = terms_dic.keys()
        self.inverted_idx['__doc_number__'] += 1
        self.inverted_idx['__total_length__'] += document.get_unique_words_amount()
        for key in key_list:
            if len(key) == 0: continue
            term_dic_value = terms_dic[key]
            if key.lower() in self.inverted_idx and key.isupper():
                key = key.lower()
            if key.upper() in self.inverted_idx and key.islower():
                self.changeTerm(key.upper(), key.lower())
            if len(key) == 0:
                continue
            if len(key[0]) > 1:
                continue
            garage_number = self.__get_file_ID(key)
            if key in self.inverted_idx:
                updating_inverted_index = (
                    self.inverted_idx[key][0] + term_dic_value, self.inverted_idx[key][1] + 1)
            else:
                updating_inverted_index = (term_dic_value, 1)

            self.inverted_idx[key] = updating_inverted_index

        entites_dict = document.entites
        for entity in entites_dict.keys():
            if len(entity) == 0: continue
            if entity in self.inverted_idx:
                garage_number = self.__get_file_ID(entity)
                self.inverted_idx[entity] = (
                    self.inverted_idx[entity][0] + entites_dict[entity], self.inverted_idx[entity][1] + 1)
            else:
                if entity in self.__pending_list:
                    garage_number = self.__get_file_ID(entity)
                    self.inverted_idx[entity] = (
                        entites_dict[entity] + self.__pending_list[entity], 2)
                else:
                    self.__pending_list[entity] = entites_dict.get(entity)

        self.__document_list.append(document)

    def building_posting_file(self):
        for document in self.__document_list:
            self.__building_posting_file(document.get_twit_id(), document.term_doc_dictionary, document.entites,
                                         document.get_max_tf(), document.get_unique_words_amount())

        self.__document_list = None

    def __building_posting_file(self, twit_id, term_doc, entite_doc, max_tf, unique_words):
        """
        this function iterate document object and extract data to posting files - separated by the first latter of
        the term an addition it iterate on entity that collected in parse stage and check if its suppose to join to
        the inverted index (if entity exist more then once its join to inverted index) /n This function fill a
        dictionary called grage - each letter have a representation in this dictionary - when some dictionary contain
        more then limit (defined in the constructor) its call to function that drain all the latter data to that file
        its belong to - when we cross this limit the function send this dictionary and the 4 biggest dictionary to drain
        :param document: document

        object :return: None
        """
        terms_dic = term_doc
        key_list = terms_dic.keys()
        # calculate data for using bm25 similarity
        weight_doc_by_tf_idf = {}
        for key in key_list:
            if len(key) == 0: continue
            if key not in self.inverted_idx:
                if key.upper() in self.inverted_idx:
                    token = key.upper()
                elif key.lower() in self.inverted_idx:
                    token = key.lower()
            else:
                token = key
            tf_ij = terms_dic[key] / unique_words
            weight_doc_by_tf_idf[key] = (tf_ij , math.log10(self.inverted_idx['__doc_number__'] / self.inverted_idx[token][1]))

        for key in entite_doc:
            if len(key) == 0: continue
            if key not in self.inverted_idx:
                continue
            tf_ij = entite_doc[key] / unique_words
            weight_doc_by_tf_idf[key] = ( tf_ij, math.log10(self.inverted_idx['__doc_number__'] / self.inverted_idx[key][1]))


        # starting insert data to posting garage

        for key in key_list:
            if len(key) == 0: continue
            token = key
            if key not in self.inverted_idx:
                if key.upper() in self.inverted_idx:
                    token = key.upper()
                elif key.lower() in self.inverted_idx:
                    token = key.lower()
            else:
                token = key

            garage_name = self.__get_file_ID(token)
            relevant_dict = self.__garage_dict[garage_name]
            if token in relevant_dict:
                relevant_dict[token].append(
                    (twit_id, unique_words, weight_doc_by_tf_idf[key][0], weight_doc_by_tf_idf[key][1]))
            else:
                relevant_dict[token] = [
                    (twit_id, unique_words, weight_doc_by_tf_idf[key][0], weight_doc_by_tf_idf[key][1])]

            self.inverted_idx[token] = (
                self.inverted_idx[token][0], self.inverted_idx[token][1])

        ###--- handle pending list - entites ---###
        entites_dict = entite_doc
        for entity in entites_dict:
            if len(entity) == 0: continue
            garage_name = self.__get_file_ID(entity)
            if entity not in self.inverted_idx:
                continue
            else:
                if entity in self.__garage_dict[garage_name]:
                    self.__garage_dict[garage_name][entity].append(
                        (twit_id,unique_words, weight_doc_by_tf_idf[entity][0],
                         weight_doc_by_tf_idf[entity][1]))
                else:
                    self.__garage_dict[garage_name][entity] = [(
                        twit_id, unique_words, weight_doc_by_tf_idf[entity][0], weight_doc_by_tf_idf[entity][1])]

        ###--- done handle entity ---###

    def __get_file_ID(self, key):
        """
        if term in inverted index its already have a posting file number \n
        otherwise- its will get a number and push it into the data that saved in inverted index
        :param key: the term we want to get his posting file number
        :return: numbert that represent the posting file that 'key' saving in
        """
        return 0

    def empty_pendling(self):
        """
        empty pendling list
        :return: None
        """
        self.__pending_list = {}

    def get_posting_file_part_c_requirment(self):
        """
        In part C of the assigment we require to change our indexer to export one posting file because the corpus we
        testing on became very small\n
        So this function using only in part C testing. \n
        :return: Single posting file data.
        """
        return self.__garage_dict[0]

    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        file_stream=open(fn,'rb')
        data=pickle.load(file_stream)
        file_stream.close()
        self.inverted_idx=data
        # self.postingDict=data[1]

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, fn):
        """
        Saves a pre-computed index (or indices) so we can save our work.
        Input:
              fn - file name of pickled index.
        """
        file_stream=open(fn,'wb')
        # pickle.dump((self.inverted_idx,self.postingDict),file_stream)
        pickle.dump(self.inverted_idx,file_stream)
        file_stream.close()


    # feel free to change the signature and/or implementation of this function
    # or drop altogether.
    def _is_term_exist(self, term):
        """
        Checks if a term exist in the dictionary.
        """
        return term in self.postingDict

    # feel free to change the signature and/or implementation of this function
    # or drop altogether.
    def get_term_posting_list(self, term):
        """
        Return the posting list from the index for a term.
        """
        return self.postingDict[term] if self._is_term_exist(term) else []
