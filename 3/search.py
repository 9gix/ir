#!/usr/bin/python3

import nltk
import pickle
import argparse
import math
import collections

porter = nltk.PorterStemmer()

def postings_opener(postings_file, dictionary):
    """Retriever helper function (wrapper)"""
    def postings(term):
        with open(postings_file, 'rb') as f:
            try:
                f.seek(dictionary[term][1])
            except KeyError:
                doc_set = set()
            else:
                doc_set = pickle.load(f)
        return doc_set
    return postings

class SearchEngine(object):
    def __init__(self, dictionary, doc_set, doc_length, postings):
        self.dictionary = dictionary
        self.doc_set = doc_set
        self.postings = postings
        self.doc_length = doc_length


    def generate_query_weighting(self, query):
        tf = collections.Counter(query)
        calculate_log_tf = lambda q: (
                1 + tf[q]
                if tf[q] > 0 else 0)
        calculate_idf = lambda q: (
                math.log10(len(self.doc_set)/self.dictionary[q][0])
                if q in self.dictionary else 0)
        calculate_weight = (
                lambda q: calculate_log_tf(q) * calculate_idf(q))

        weighted_qt = [
            (term, calculate_weight(term)) for term in query]
        sum_of_square_weight = sum(weight**2 for term,weight in weighted_qt)
        normalized_wqt = map(
            lambda x: (x[0], x[1]/math.sqrt(sum_of_square_weight)),
            weighted_qt)
        return normalized_wqt

    def tokenize(self, text):
        terms = nltk.word_tokenize(text)
        terms = map(lambda term: porter.stem(term).lower(), terms)
        return terms

    def search(self, query, result_count):
        """Perform the search queries"""
        query = list(self.tokenize(query))
        query_terms = self.generate_query_weighting(query)

        scores = collections.defaultdict(int)
        for query_term, term_query_weight in query_terms:
            for doc in self.postings(query_term):
                term_doc_id, term_doc_weight = doc
                scores[term_doc_id] += term_doc_weight * term_query_weight

        normalized_scores = {score/math.sqrt(self.doc_length[doc]
            for doc, score in score.items}

        sorted_scores = sorted(normalized_scores.items(), 
                key=lambda x: x[1], reverse=True)
        return sorted_scores[:result_count]


def main(dict_file, postings_file, queries_file, output_file):
    # Load Dictionary
    with open(dict_file, 'rb') as f:
        doc_id_set, doc_length, dictionary = pickle.loads(f.read())

    postings = postings_opener(postings_file, dictionary)

    # Read queries
    with open(queries_file) as f:
        queries = f.readlines()

    # Perform Queries
    results = []
    for query in queries:
        search_engine = SearchEngine(dictionary, doc_id_set, doc_length, postings)
        result = search_engine.search(query, result_count=10)
        results.append(result)

    # Store Result
    with open(output_file, 'w') as f:
        for result in results:
            result = (int(i[0]) for i in result)
            x = ' '.join(str(doc_id) for doc_id in result)
            f.write(x + '\n')
        

def _getCommandArgs():
    parser = argparse.ArgumentParser(description='Document search')
    parser.add_argument('-d', dest='dict_file', required=True,
            help='directory file')
    parser.add_argument('-p', dest='postings_file', required=True,
            help='postings file')
    parser.add_argument('-q', dest='queries_file', required=True,
            help='queries file')
    parser.add_argument('-o', dest='output_file', required=True,
            help='search result output file')
    return parser.parse_args()

if __name__ == '__main__':
    args = _getCommandArgs()
    main(args.dict_file, args.postings_file,
        args.queries_file, args.output_file)
