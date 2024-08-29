"""
This script implements a Naive Bayes classifier to detect spam emails using the vectorized data from `vectorizator.py`.
It uses the `MultinomialNB` class from `sklearn` to train the model and predict the labels for the test set. It also
calculates the accuracy, precision, recall, and F1 score for each dictionary dimension and plots the ROC curve for each
dimension.
"""
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

from vectorizator import vectorization

performance_metrics = []

dict_dim = [100, 500, 1000, 2000, 3000]

# TODO: Move into function and call it for each dictionary dimension. Useful for testing different dimensions.
# TODO: Function for vectorization and another one for plot. In the end also a function to evaluate various metrics.
# TODO:  Take in account also support that means the normal distribution of the dataset
# TODO: Modify the ROC to have a single curve that plot the different dictionary dimensions and dd threshold
for dim in dict_dim:
    # Vectorize the training and testing datasets
    df_train = vectorization('../dataset/train-mails', dim)
    df_test = vectorization('../dataset/test-mails', dim)

    # Separate features and labels for training and test data
    X_train = df_train.drop(columns=['is_spam'])
    y_train = df_train['is_spam']

    X_test = df_test.drop(columns=['is_spam'])
    y_test = df_test['is_spam']

    # Initialize and train the Multinomial Naive Bayes model
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Predict the labels for the test set
    y_test_pred = clf.predict(X_test)

    # Generate and plot the ROC curve among the different dictionary dimensions
    fpr, tpr, _ = roc_curve(y_test, y_test_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'Dictionary Dimension = {dim} (area = {roc_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    # print the accuracy, precision, recall and f1 score
    accuracy = clf.score(X_test, y_test)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    # Store the metrics in the DataFrame
    performance_metrics.append({
        'Dictionary Dimension': dim,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

performance_metrics_df = pd.DataFrame(performance_metrics, index=dict_dim)

print(performance_metrics_df)
plt.savefig('roc_curve_naive_bayes.png')
plt.show()
