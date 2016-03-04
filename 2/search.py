#!/usr/bin/python3

import nltk
import pickle
import argparse

porter = nltk.PorterStemmer()

# Constant Declaration
LEFT_PAR = '('
RIGHT_PAR = ')'
PARENTHESES = (LEFT_PAR, RIGHT_PAR)

AND_OP = 'AND'
OR_OP = 'OR'
NOT_OP = 'NOT'
OPERATORS = (AND_OP, OR_OP, NOT_OP)

PRECEEDENCE = {
    NOT_OP: 3,
    AND_OP: 2,
    OR_OP: 1,
}

# Helper methods
is_operator = lambda token: token in OPERATORS
is_parentheses = lambda token: token in PARENTHESES
is_operand = lambda token: not (is_operator(token) or is_parentheses(token))
is_lower_preceedence = lambda x, y: PRECEEDENCE[x] <= PRECEEDENCE[y]



def postings_retriever(postings_file, dictionary):
    """Retriever helper function (wrapper)"""
    def get_doc_set(term):
        with open(postings_file, 'rb') as f:
            try:
                f.seek(dictionary[term][1])
            except KeyError:
                doc_set = set()
            else:
                doc_set = pickle.load(f)
            
        return doc_set
    return get_doc_set


def shunting(tokens):
    """Shunting algorithms will convert infix notation
    into reverse polish notation"""
    output_queue, operator_stack = [], []
    for token in tokens:
        if is_operand(token):
            output_queue.append(token)
        elif is_operator(token):
            while (operator_stack and not is_parentheses(operator_stack[-1]) and is_lower_preceedence(token, operator_stack[-1])):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == LEFT_PAR:
            operator_stack.append(token)
        elif token == RIGHT_PAR:
            while (operator_stack and operator_stack[-1] != LEFT_PAR):
                output_queue.append(operator_stack.pop())
            operator_stack.pop()

    while operator_stack:
        output_queue.append(operator_stack.pop())

    return output_queue

def search(query_tokens, universal_doc, docset):
    """Perform the search queries"""
    result_doc_set = set()
    operand_stack = []
    query_tokens = query_tokens[::-1]

    while (len(query_tokens) >= 1):
        tok = query_tokens.pop()
        if is_operand(tok):
            operand_stack.append(docset(porter.stem(tok).lower()))
        else:
            if tok == NOT_OP:
                operand_stack.append(universal_doc - operand_stack.pop())
            elif tok == AND_OP:
                o1 = operand_stack.pop()
                o2 = operand_stack.pop()
                operand_stack.append(o1 & o2)
            elif tok == OR_OP:
                o1 = operand_stack.pop()
                o2 = operand_stack.pop()
                operand_stack.append(o1 | o2)
    return operand_stack[0]


def main(dict_file, postings_file, queries_file, output_file):
    # Load Dictionary
    with open(dict_file, 'rb') as f:
        doc_id_set, dictionary = pickle.loads(f.read())

    doc_set = postings_retriever(postings_file, dictionary)

    # Read queries
    with open(queries_file) as f:
        queries = f.readlines()


    # Perform Queries
    results = []
    for query in queries:
        tokens = nltk.word_tokenize(query)
        query_tokens = shunting(tokens)
        result = search(query_tokens, doc_id_set, doc_set)
        results.append(result)

    # Store Result
    with open(output_file, 'w') as f:
        for result in results:
            result = (int(i) for i in result)
            x = ' '.join(str(doc_id) for doc_id in sorted(result))
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
