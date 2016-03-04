#!/usr/bin/python3

import argparse
import os
import nltk
import collections
import pickle


def extract(document):
    """Extraction of terms in the document"""
    sentences = []
    for sentence in nltk.sent_tokenize(document):
        sentences.append(nltk.word_tokenize(sentence))
    return sentences

    

def main(index_dir, dict_file, postings_file):

    dictionary = {}
    postings = collections.defaultdict(set)
    doc_id_set = set()

    for filename in os.listdir(index_dir):
        doc_id = filename
        filepath = os.path.join(index_dir, filename)
        with open(filepath) as f:
            document = f.read()

        sentences = extract(document)
        porter = nltk.PorterStemmer()

        for sentence in sentences:
            tokens = [porter.stem(token) for token in sentence]
            tokens = [token.lower() for token in tokens]
            for token in tokens:
                postings[token].add(doc_id)
                doc_id_set.add(doc_id)

    print("Saving Index")

    with open(postings_file, 'wb') as f:
        for term, doc_set in postings.items():
            posting_offset = f.tell()
            dictionary[term] = (len(doc_set), posting_offset)
            pickle.dump(doc_set, f)


    with open(dict_file, 'wb') as f:
        pickle.dump((doc_id_set, dictionary), f)





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
