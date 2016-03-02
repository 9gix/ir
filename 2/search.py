#!/usr/bin/python3

import nltk
import pickle


class QueryParser(object):
    pass



def main(dict_file, postings_file, queries_file, output_file):
    with open(queries_file) as f:
        queries = f.readlines()

    with open(dict_file, 'rb') as f:
        dictionary = pickle.loads(f.read())

    for query in queries:
        parser = QueryParser()
        query_tokens = parser.parse(query)
        for query_token in query_tokens:
            pass
        



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
