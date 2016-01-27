#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import argparse


"""

Language Model Data Structure:
    - _dictionary: 
        {
            <token-1>: <count>
            <token-2>: <count>
            <token-3>: <count>
            <token-4>: <count>
        }
    - token_count

3 language model object:
    - malay language model
    - indonesia language model
    - tamil language model


"""

    
def test_LM(in_file, out_file, lm):
    """
    test the language models on new URLs
    each line of in_file contains an URL
    you should print the most probable label for each URL into out_file
    """
    print("testing language models...")
    # This is an empty method
    # Pls implement your code in below
    test_data = []
    with open(in_file) as f:
        for line in f:
            test_data.append({
                'sentence': line,
            })

    for data in test_data:
        data['predicted'] = lm.predict(line)

    with open(out_file, 'w') as f:
        for data in test_data:
            print(data)
            f.write("{predicted} {sentence}\n".format(**data))

class Tokenizer(object):
    pass

class WordTokenizer(Tokenizer):
    pass

class CharacterTokenizer(Tokenizer):
    def __init__(self, ngram=4):
        self.ngram = ngram

    def tokenize(self, sentence, pad=True):
        if pad:
            sentence = self._pad(sentence.lower(), self.ngram-1)
        token_list = []
        for i in range(0, len(sentence) - (self.ngram-1)):
            token_list.append(sentence[i:i+self.ngram])
        return token_list

    def _pad(self, sentence, pad_size, pad_char='\0'):
        return pad_char * pad_size + sentence + pad_char * pad_size


class LanguagePredictor(object):
    def __init__(self, language_model_table, tokenizer):
        self._language_model_dict = language_model_table
        self.tokenizer = tokenizer

    def predict(self, sentence):
        prediction_dict = {}
        for lang in self._language_model_dict.keys():
            prediction_dict[lang] = 1

        for token in self.tokenizer.tokenize(sentence):
            for lang in self._language_model_dict.keys():
                prediction_dict[lang] *= (
                    self._language_model_dict[lang][token] /
                    self._language_model_dict[lang].token_count)
        
        result = max(prediction_dict.keys(), key=(
            lambda lang: prediction_dict[lang]))
        #print(prediction_dict[result])
        if prediction_dict[result] == 0:
            return 'other'
        return result


class LanguageModel(object):
    """
    an abstraction of a language dictionary.
    The dictionary will contains:
    - <token> as the key
    - <count> as the value

    aside from the dictionary, the language model also 
    track the sum count of all token.
    """
    def __init__(self, language, tokenizer):
        self._dict = {}
        self.language = language
        self.token_count = 0
        self.tokenizer = tokenizer
        self.smooth_value = 0

    def learn(self, sentence):
        for token in self.tokenizer.tokenize(sentence):
            self[token] += 1
            self.token_count += 1

    def smoothing(self, value=0):
        self.smooth_value = value
        for token in self._dict.keys():
            self[token]+=value
            self.token_count+=value

    def __getitem__(self, token):
        """Retrieve the token count from the dictionary"""
        return self._dict.get(token, self.smooth_value)


    def __setitem__(self, token, value):
        """Set the token count into the dictionary"""
        self._dict[token] = value
        
    def __repr__(self):
        return "{data} - total: {total}".format(
                data=self._dict, total=self.token_count)

def build_LM(input_file_b, tokenizer):
    """
    build language models for each label
    each line in in_file contains a label
    and an URL separated by a tab(\t)

    return dictionary with the following formats:
    {
        <language>: <language-model>,
        <language>: <language-model>,
        <language>: <language-model>,
    }
    """
    print('building language models...')

    with open(input_file_b) as f:
        sample_data = f.readlines()

    language_models = {}
    for line in sample_data:
        language, sentence = line.split(' ', 1)

        if language not in language_models:
            language_models[language] = LanguageModel(language, tokenizer)

        language_models[language].learn(sentence)

    for lang, language_model in language_models.items():
        language_model.smoothing(1)

    return language_models


def main(input_file_b, input_file_t, output_file):
    tokenizer = CharacterTokenizer(ngram=4)
    language_models = build_LM(input_file_b, tokenizer)
    print(language_models)
    predictor = LanguagePredictor(language_models, tokenizer)

    with open(input_file_t) as f:
        test_data = f.readlines()

    result = []
    for line in test_data:
        result.append(predictor.predict(line))

    with open(output_file, 'w') as f:
        for i in range(len(test_data)):
            f.write("{predicted} {sentence}".format(**{
                'predicted':result[i],
                'sentence': test_data[i],
            }))

    
def getCommandArgs():
    parser = argparse.ArgumentParser(description='Detect Language')
    parser.add_argument('-b', metavar='input-file-for-building-LM',
            type=str, help='input file for building the language model',
            dest='input_file_b', required=True)
    parser.add_argument('-t', metavar='input-file-for-testing-LM',
            type=str, help='input file for testing the language',
            dest='input_file_t', required=True)
    parser.add_argument('-o', metavar='output-file',
            type=str, help='output file of the language prediction',
            dest='output_file', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = getCommandArgs()
    main(args.input_file_b, args.input_file_t, args.output_file)
