#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import argparse
import pprint
import math


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

class Tokenizer(object):
    """Base class for all tokenizer type,
    Any child class must implement tokenize method
    """
    def tokenize(self, sentence):
        """
        @sentence the line will be tokenize
        """
        raise NotImplementedError()

class WordTokenizer(Tokenizer):
    def tokenize(self, sentence):
        """Tokenize word by splitting any whitespace"""
        return sentence.strip().split()

class CharacterTokenizer(Tokenizer):
    def __init__(self, ngram=4, pad=False):
        self.ngram = ngram
        self.pad = pad

    def replace_digits(self, sentence, digits_representation='0'):
        """replace all digits into `0`, 
        so we can treat any `0` as any number"""
        return re.sub(r'\d+', digits_representation, sentence)

    def normalize_spaces(self, sentence):
        """remove redundant white spaces"""
        return ' '.join(sentence.strip().split())

    def _pad(self, sentence, pad_size, pad_char='\0'):
        """Pad left and right most with `\0` (null character),
        @pad_size the amount of character padded"""
        return pad_char * pad_size + sentence + pad_char * pad_size

    def tokenize(self, sentence):
        """N-gram tokenization method
        @sentence the line will be tokenize into char of ngram
        @pad will create add null character on the left and right.
        """
        sentence = self.normalize_spaces(sentence)
        sentence = self.replace_digits(sentence)
        if self.pad:
            sentence = self._pad(sentence.lower(), self.ngram-1)
        token_list = []
        for i in range(0, len(sentence) - (self.ngram-1)):
            token_list.append(sentence[i:i+self.ngram])
        return token_list


class LanguagePredictor(object):
    """Predictor handles the prediction based on the language model given"""

    # static counter for debugging purpose
    prediction_index = 1
    def __init__(self, language_model_dict, tokenizer):
        self._language_model_dict = language_model_dict
        self.tokenizer = tokenizer

    def predict(self, sentence):
        """Predict the probability of a sentence
        based on the language model registered"""
        prediction_dict = {}
        for lang in self._language_model_dict.keys():
            prediction_dict[lang] = 0

        tokens = self.tokenizer.tokenize(sentence)

        # For each token, it will sum the total count
        for token in tokens:
            most_used_lang = max(
                    self._language_model_dict,
                    key=lambda lang: self._language_model_dict[lang][token])
            if (self._language_model_dict[most_used_lang][token] > 1):
                for lang, lang_model in self._language_model_dict.items():
                # Do the computation in log because of the small floating point
                    prediction_dict[lang] += math.log(lang_model[token])
                    prediction_dict[lang] -= math.log(lang_model.token_count)
        

        # [(lang1, 0.01), (lang2, 0.005), ...]
        pred = list(map(
            lambda lang: (
                lang, 
                prediction_dict[lang]
            ), prediction_dict.keys()))

        # Get the Highest prediction
        result = max(pred, key=(lambda x: x[1]))

        # Unknown language when probability is 0
        if result[1] == 0:
            predicted_language = 'other'
        else:
            predicted_language = result[0]

        #print(sentence)
        #print(LanguagePredictor.prediction_index, predicted_language, pred)

        # simple counter for debugging purpose
        LanguagePredictor.prediction_index+=1
        return predicted_language

class LanguageModel(object):
    """
    language model is an abstraction of a language dictionary.
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

    def train(self, sentence):
        """Tokenize and add each token into the dictionary counter"""
        for token in self.tokenizer.tokenize(sentence):
            self[token] += 1
            self.token_count += 1

    def smoothing(self, value=0):
        """increase the whole dictionary counter by value"""
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
        """Debugging message"""
        data = pprint.pformat(self._dict, indent=2)
        return "{language} - total: {total}".format(
                language=self.language, total=self.token_count)

def build_LM(input_file_b, tokenizer=CharacterTokenizer(ngram=4)):
    """
    build language models for each label
    each line in in_file contains a label
    and an URL separated by a tab(\t)

    return dictionary with the following formats:
    {
        <language-A>: <language-model>,
        <language-B>: <language-model>,
        <language-C>: <language-model>,
    }
    """
    print('Training language models...')

    with open(input_file_b) as f:
        sample_data = f.readlines()

    language_models = {}
    for line in sample_data:
        language, sentence = line.split(' ', 1)

        if language not in language_models:
            language_models[language] = LanguageModel(language, tokenizer)

        language_models[language].train(sentence)

    # Smoothing
    for lang, language_model in language_models.items():
        language_model.smoothing(1)

    # Print total tokens
    #pprint.pprint(language_models)

    return language_models

def test_LM(in_file, out_file, lm, tokenizer=CharacterTokenizer(ngram=4)):
    """
    predict the language of each in_file lines.
    """
    print("Predicting language...")
    predictor = LanguagePredictor(lm, tokenizer)

    # Read test file into test_data
    with open(in_file) as f:
        test_data = f.readlines()

    # Predict the language of test_data
    result = []
    for line in test_data:
        result.append(predictor.predict(line))

    # Save the predicted test_data languages
    with open(out_file, 'w') as f:
        for i in range(len(test_data)):
            f.write("{predicted} {sentence}".format(**{
                'predicted': result[i],
                'sentence': test_data[i],
            }))

def main(input_file_b, input_file_t, output_file):
    """Train and Predict the language"""
    tokenizer = CharacterTokenizer(ngram=4, pad=False)
    #tokenizer = WordTokenizer()
    language_models = build_LM(input_file_b, tokenizer)
    test_LM(input_file_t, output_file, language_models, tokenizer)

    
def getCommandArgs():
    """Handling arguments for the command line"""
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
