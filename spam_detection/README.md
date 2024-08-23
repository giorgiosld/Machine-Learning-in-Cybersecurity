# Spam Detection

This directory contains the implementation of a spam email classifier developed as part of the "Machine Learning in
Cybersecurity" course at Reykjavík University during the Fall Semester 2024.

## Project Overview
This project involves implementing a spam email classifier using Python and scikit-learn. The classifier is trained to 
identify spam emails based on textual content.

## Tasks
1. **Data Preprocessing**:
   + Removal of stop-words and lemmatization.
   + Creating a dictionary of the 2000 most frequent words after filtering out non-words.
2. **Feature Extraction**:
   + Transform each email into a word count vector based on the dictionary.
3. **Model Training**: 
   + Train a Naive Bayes classifier `(MultinomialNB from sklearn.naive_bayes)`.
4. **Model Evaluation**: 
   + Test the trained classifier on a test set and evaluate its performance.

## Usage
1. Preprocess the data:
   ```
   python preprocess.py
   ```
2. Train the classifier:
   ```
   python vectorize.py
   ```
3. Evaluate the classifier:
   ```
   python analyze_data.py
   ```

## Dataset
+ The dataset used to train the model is the [Lingspam Public Dataset](train-mails).
+ The dataset used to evaluate the model is the [Lingspam Public Dataset](test-mails).
