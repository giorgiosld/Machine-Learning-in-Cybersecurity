"""
This script processes emails from a specified directory by first applying custom preprocessing from `preprocess.py`,
then vectorize the content using `CountVectorizer` from `sklearn`, and finally converts the vectorized data into a
pandas DataFrame, including an 'is_spam' column to label each email as spam (1) or not (0).
"""
import os
from utils.preprocess import preprocess_data

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# create a vectorizer object to transform the data into a word count vector
def vectorize_data(data, vocabulary):
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    X = vectorizer.fit_transform(data)
    return X, vectorizer

# function to manage the emails in the directory
def manage_emails(directory):
    emails = []
    file_names = []
    files = os.listdir(directory)
    for file in files:
        with open(os.path.join(directory, file), 'r') as f:
            data = f.read()
            emails.append(data)
            file_names.append(file)
    return emails, file_names

# function to convert the word count vector into a pandas dataframe
def converter(X, feature_names, file_names):
    df = pd.DataFrame(X.toarray(), columns=feature_names, index=file_names)
    return df

# function to wrap the vectorization process to return a pandas dataframe usable in other classes
def vectorization(directory, dictionary_dim):
    # get an absolute path to avoid problem when the code run from another directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    emails_dir = os.path.abspath(os.path.join(script_dir, directory))

    emails, file_names = manage_emails(emails_dir)

    preprocess_dir = os.path.abspath(os.path.join(script_dir, '../../dataset/train-mails'))
    word_counts = preprocess_data(preprocess_dir, dictionary_dim)

    vocabulary = [word for word, count in word_counts]
    X, vectorizer = vectorize_data(emails, vocabulary)
    df = converter(X, vectorizer.vocabulary, file_names)
    df['is_spam'] = [1 if file.startswith('spm') else 0 for file in file_names]
    return df

if __name__ == "__main__":
    vectorization('../dataset/train-mails')