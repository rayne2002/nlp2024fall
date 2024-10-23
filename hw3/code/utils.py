import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            synonyms.add(synonym)
    return list(synonyms)

import random

keyboard_map = {
    'a': ['s'],
    'b': ['v', 'n'],
    'c': ['x'],
    'd': ['f'],
    'e': ['r'],
    'f': ['g', 'd'],
    'g': ['f', 'h'],
    'h': ['g', 'j'],
    'i': ['o', 'u'],
    'j': ['h', 'k'],
    'k': ['j', 'l'],
    'l': ['k'],
    'm': ['n'],
    'n': ['b', 'm'],
    'o': ['i', 'p'],
    'p': ['o'],
    'q': ['w'],
    'r': ['e', 't'],
    's': ['a', 'w'],
    't': ['r', 'y', 'g', 'h'],
    'u': ['i', 'o'],
    'v': ['b'],
    'w': ['q', 's'],
    'x': ['c'],
    'y': ['t', 'u'],
    'z': []
}

def introduce_typos(word):
    typo_word = list(word.lower())
    words_with_typos = []

  
    for _ in range(len(typo_word)):
        index_to_change = random.randint(0, len(typo_word) - 1)
        original_char = typo_word[index_to_change]

        if original_char in keyboard_map and keyboard_map[original_char]:
            typo_word[index_to_change] = random.choice(keyboard_map[original_char])
        else:
            typo_word[index_to_change] = random.choice('abcdefghijklmnopqrstuvwxyz')

        words_with_typos.append(''.join(typo_word))

    return random.choice(words_with_typos)

def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    tokens = word_tokenize(example["text"])
    new_tokens = []
    synonym_replaced = False

    for token in tokens:
        if len(token) >= 4 and token.lower() not in [".", ",", "!", "?", ";", ":", " "] and not synonym_replaced:
            synonyms = get_synonyms(token)
            if synonyms:
                new_token = random.choice(synonyms)
                synonym_replaced = True
            else:
                new_token = token
        else:
            new_token = token

        if random.random() < 0.1:
            new_token = introduce_typos(new_token)

        new_tokens.append(new_token)

    example["text"] = ' '.join(new_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example

