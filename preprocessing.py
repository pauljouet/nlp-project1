import unidecode
import re
import os
from collections import defaultdict
from typing import List
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from gensim.models.word2vec import Word2Vec

DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIR, "data/filtered_openfoodfacts.csv")
MODEL_PATH = os.path.join(DIR, "models/word_embedding.model")
TOKENS_PATH = os.path.join(DIR, "data/tokens_list.txt")

INFIX_DELIMITERS = {
    ':': ' ', 
    '-': ' ', 
    ',': ' ', 
    ';':' ', 
    '.':' ', 
    "'": ' ',
    "/": ' ',
    "%": ' ',
    'œ': 'oe'}
# For the first version (where we keep the delimiters)
PREFIX_DELIMITERS = ['(', '[']
SUFFIX_DELIMITERS = [')', ']', '.', ',', ';', ':']

# region Version 1 functions
def remove_percentage(text) -> list:
    """This function remove any percentage from a string"""
    return re.sub(r'(\d+( |.)?(\d+)?( )?%)', "", text)

def flatten_tokens(tokens):
    flat_tokens = []
    for token in tokens:
        if type(token) == list:
            flat_tokens += token
        else:
            flat_tokens.append(token)
    return flat_tokens

def split_punct_prefix(tokens):
    for i in range(len(tokens)):
         if len(tokens[i]) > 1:   
            if tokens[i][0] in PREFIX_DELIMITERS:
                tokens[i] = [tokens[i][0], tokens[i][1:]]
    # print(tokens)
    return flatten_tokens(tokens)

def split_punct_suffix(tokens):
    # print("len:", len(tokens), "suffix:", tokens)
    for i in range(len(tokens)):
        # print(tokens[i])
        if len(tokens[i]) > 1:
            if tokens[i][-1] in SUFFIX_DELIMITERS:
                tokens[i] = [tokens[i][:-1], tokens[i][-1]]
    return flatten_tokens(tokens)

def split_puncts(tokens):
    return split_punct_suffix(split_punct_prefix(tokens))

def tokenizer_v1(ingr: str):
    ingr1 = remove_special_characters(ingr.lower())
    ingr2 = remove_accent(ingr1)
    ingr3 = remove_percentage(ingr2)
    ingr4 = replace_infix(ingr3)
    ingr_list = remove_numbers(ingr4.split())
    ingr_list1 = list(filter(lambda tok: tok != '', ingr_list))
    return split_puncts(ingr_list1)

# region Tests V1

def test_remove_percentage():
    print(remove_percentage("acide15.1%, nitrate15,1%, ammoniac 15.1%, levure 15,1%, banane 15.6 %, orange 20,8 %, fruit 25 %, cassis 15.7%ù kiwi 7815%, trucmuche 15 . 1 % autre 15 , 1 % jaipludinspi 15 . 1% gngn 15 , 1% "))

# endregion
# endregion

# region Version 2 functions

def replace_infix(text):
    for delimiter in INFIX_DELIMITERS.keys():
        text = text.replace(delimiter, INFIX_DELIMITERS[delimiter])
    return text

def remove_special_characters(text: str):
    text = text.translate(str.maketrans("", "", r"*•&\=+_~#²<>!?"))
    return text

def remove_accent(accented_string) -> str:
    return unidecode.unidecode(accented_string)

def remove_numbers(tokens):
    return list(filter(lambda x: not x.isnumeric(), tokens))

def tokenizer_v2(ingr: str):
    ingr = ingr.lower().translate(str.maketrans("", "", r"*•&\/=+_~#²<>!?{}()[]."))
    ingr = remove_accent(ingr)
    ingr = remove_percentage(ingr)
    ingr = replace_infix(ingr)
    toks = "".join(filter(lambda x: not x.isdigit(), ingr)).split()
    toks = list(filter(lambda tok: len(tok) > 2, toks))
    stemmer = SnowballStemmer(language = "french")
    return list(map(stemmer.stem, toks))

# region Tests V2

def test_remove_special_characters():
    print(remove_special_characters("ndgn*dbs•z&\\\/=kvcxng+_sbdbfs~#²sbbs<sb> sbcdjbokdsc"))

def test_remove_numbers():
    print(remove_numbers("dfhdi 15 dijv 75 vjsoi 14.7, dijviv e205, dvov15, 14, 14,7    19, kd".split()))

# endregion
# endregion

def tokenize(df) -> List[List[str]]:
    tokenized = []
    for i in range(len(df)):
        tokenized.append(tokenizer_v2(df.iloc[i]['ingredients_text']))
    return tokenized

def clean_tokens(tokens_list: List[List[str]]) -> List[List[str]]:
    """Deletes tokens which appear less than 50 times in the dataset. This removes irrelevant words or mistakes, 
    however we lose more information than if we used a spell checker.
    Our first strategy was to match the less common tokens with the most common ones using a Levanshtein distance of the chars."""
    ingredients = defaultdict(int)
    for ingr_list in tokens_list:
        for ingr in ingr_list:
            ingredients[ingr] += 1
    return list(map(lambda y: list(filter(lambda x: ingredients[x] > 50, y)), tokens_list))

def word_embedding(tokens_list: List[List[str]]):
    """Returns a gensim model for the word embedding using Word2Vec"""
    class Corpus:
        """An iterator that yields sentences (lists of str)."""
        def __iter__(self):
            for ingredient_list in tokens_list:
                yield ingredient_list
    sentences = Corpus()
    return Word2Vec(sentences=sentences)

def save_tokens_list(tokens_list, path= TOKENS_PATH):
    with open(path, 'w') as f:
        for ingr_list in tokens_list:
            f.write(','.join(ingr_list))
            f.write('\n')

def get_tokens_list():
    if os.path.exists(TOKENS_PATH):
        tokens_list = []
        with open (TOKENS_PATH, 'r') as f:
            for line in f:
                tokens_list.append(line.split(','))
        return tokens_list
    return None
    

def preprocess(df):
    if os.path.exists(TOKENS_PATH):
        tokens_list = get_tokens_list()
    else:
        tokens_list = clean_tokens(tokenize(df))
        save_tokens_list(tokens_list)

    if os.path.exists(MODEL_PATH):
        model = Word2Vec.load(MODEL_PATH)
    else:
        model = word_embedding(tokens_list)
    if not os.path.exists(MODEL_PATH):
        model.save(MODEL_PATH)
    
    return tokens_list, model

if __name__ == "__main__":
    print(tokenize(pd.read_csv(DATA_PATH, encoding= 'utf-8', delimiter= '\t'))[:10])