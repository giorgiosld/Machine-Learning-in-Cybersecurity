"""
This script preprocesses data from a directory of text files, generating a dictionary of the 2000 most common words and
 converting each file into a word count vector.
"""

import os

# function to print insights from the sorted word count dictionary
def print_insight(sorted_word_counts):
    for key, value in sorted_word_counts.items():
        print(key, ":", value)
    print("Total words:", len(sorted_word_counts))

# function to count the words in a line
def count_words(words, word_counts):
    for word in words:
        if word.isalpha():
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

# function to manage each file's lines in the directory
def manage_file(file, directory, word_counts):
    with open(os.path.join(directory, file), 'r') as f:
        next(f)
        for line in f:
            line = line.strip().lower()
            words = line.split(" ")
            count_words(words, word_counts)

# function to manage directory and files returning the sorted word count dictionary
def preprocess_data(directory, limit=2000):
    word_counts = dict()
    directory = os.path.abspath(directory)
    files = os.listdir(directory)
    for file in files:
        manage_file(file, directory, word_counts)
    sorted_word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))
    limited_word_counts = list(sorted_word_counts.items())[:limit]
    return limited_word_counts
