"""
Assignment 3 (Information Analyst Track) Group Member(s):
Qian Ying Wong, 49411619
Most of the code here is from my submission for Assignment 1, tokenization
"""
from nltk.stem import PorterStemmer


stemmer = PorterStemmer()

def stem_tokens(tokens):
    for token in tokens:
        yield stemmer.stem(token)


def tokenize_text(text):
    """
    Reads in text string and returns a list of the tokens in that text.
    """
    current_token = ""
    for char in text:
        if char.isascii() and char.isalnum():
            current_token += char.lower()
        else:
            if current_token != "":
                yield current_token
                current_token = ""

    if current_token != "":
        yield current_token


def tokenize(text):
    tokens = []
    for token in tokenize_text(text):
        tokens.append(token)

    return tokens
    

def compute_word_frequencies(tokens):
    """
    Counts the number of occurrences of each token in the token list.
    """
    frequencies = {}
    for token in tokens:
        if token in frequencies:
            frequencies[token] += 1
        else:
            frequencies[token] = 1

    return frequencies


def print_frequencies(frequencies):
    """
    Prints out the word frequency count by decreasing frequency.
    """
    sorted_items = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    for token, count in sorted_items:
        print(f"{token}\t{count}")



