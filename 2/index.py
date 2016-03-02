#!/usr/bin/python3

import argparse
import os
import nltk
import collections
import pickle


def extract(document):
    sentences = []
    for sentence in nltk.sent_tokenize(document):
        sentences.append(nltk.word_tokenize(sentence))
    return sentences

    

def main(index_dir, dict_file, postings_file):
    dictionary = {}

    postings = [] # List of set
    for filename in os.listdir(index_dir):
        doc_id = filename
        filepath = os.path.join(index_dir, filename)
        with open(filepath) as f:
            document = f.read()

        sentences = extract(document)
        porter = nltk.PorterStemmer()
        for sentence in sentences:
            tokens = [porter.stem(token) for token in sentence]
            for token in tokens:
                if token not in dictionary:
                    dictionary[token] = len(postings)
                    postings.append(set())
                postings[dictionary[token]].add(doc_id)

    with open(dict_file, 'wb') as f:
        pickle.dump(dictionary, f)

    with open(postings_file, 'wb') as f:
        pickle.dump(postings, f)




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
