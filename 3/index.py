#!/usr/bin/python3

import argparse
import os
import math
import nltk
import collections
import pickle


porter = nltk.PorterStemmer()

def extract(document):
    """Extraction of terms in the document"""
    terms = []
    for term in nltk.word_tokenize(document):
        terms.append(porter.stem(term).lower())
    return terms


def main(index_dir, dict_file, postings_file):

    dictionary = {}
    postings = collections.defaultdict(set)
    doc_id_set = set()
    doc_length = {}

    for filename in os.listdir(index_dir):
        doc_id = filename
        filepath = os.path.join(index_dir, filename)
        with open(filepath) as f:
            document = f.read()

        tokens = extract(document)
        term_freq = collections.Counter(tokens)

        log_tf = {t: 1 + math.log10(tf) for t, tf in term_freq.items()}

        for token in tokens:
            postings[token].add((doc_id, log_tf[token]))
            doc_id_set.add(doc_id)


        # Lenght of the documents for normalization
        doc_length[doc_id] = len(tokens)

    print("Saving Index")

    with open(postings_file, 'wb') as f:
        for term, doc_set in postings.items():
            posting_offset = f.tell()
            dictionary[term] = (len(doc_set), posting_offset)
            pickle.dump(doc_set, f)


    with open(dict_file, 'wb') as f:
        pickle.dump((doc_id_set, doc_length, dictionary), f)





def _getCommandArgs():
    parser = argparse.ArgumentParser(description='Document indexer')
    parser.add_argument('-i', dest='index_dir', required=True,
            help='directory of documents')
    parser.add_argument('-d', dest='dict_file', required=True,
            help='dictionary file')
    parser.add_argument('-p', dest='postings_file', required=True,
            help='postings file')
    return parser.parse_args()

if __name__ == '__main__':
    args = _getCommandArgs()
    main(args.index_dir, args.dict_file, args.postings_file)
