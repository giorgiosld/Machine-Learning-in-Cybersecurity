"""
This module is used to preprocess the data in the dataset. It counts the words in the dataset and returns the sorted word count dictionary.
"""
import os

word_counts = dict()

# function to count the words in a line
def count_words(words):
    for word in words:
        if word.isalpha() and len(word) > 1:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

# function to manage each file's lines in the directory
def manage_file(file, directory):
    with open(os.path.join(directory, file), 'r') as f:
        for line in f:
            line = line.strip().lower()
            words = line.split(" ")
            count_words(words)

# function to manage directory and files returning the sorted word count dictionary
def preprocess_data(directory, limit):
    directory = os.path.abspath(directory)
    files = os.listdir(directory)
    for file in files:
        manage_file(file, directory)
    sorted_word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))
    limited_word_counts = list(sorted_word_counts.items())[:limit]
    return limited_word_counts


if __name__ == "__main__":
    preprocess_data('../dataset/train-mails')