"""
This script implements a Naive Bayes classifier to detect spam emails using the vectorized data from `vectorizator.py`.
It uses the `MultinomialNB` class from `sklearn` to train the model and predict the labels for the test set. It also
calculates the accuracy, precision, recall, and F1 score for each dictionary dimension and plots the ROC curve for each
dimension.
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, classification_report, roc_auc_score

from utils.vectorizator import vectorization
from utils.data_analysis import prepare_data, calculate_performance_metrics, save_plot, compute_plot_roc_curve, \
    aggregate_classification_reports, display_classification_reports

# Initialize parameters used for the experiment
performance_metrics = []
dict_dim = [100, 500, 1000, 2000, 3000]
reports = {}

# The dictionary dimension is not passed as a parameter because it is a global variable used in evaluator.py
def train_best_model(X_train, y_train, X_test, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)
    y_test_prob = clf.predict_proba(X_test)[:, 1]

    # Calculate performance metrics
    report = classification_report(y_test, y_test_pred, output_dict=True)
    recall = report['1']['recall']

    # Calculate the ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = roc_auc_score(y_test, y_test_prob)

    return recall, fpr, tpr, roc_auc

def run_naive_bayes():

    for dim in dict_dim:

        # Vectorize the training and testing datasets
        df_train = vectorization('../../dataset/train-mails', dim)
        df_test = vectorization('../../dataset/test-mails', dim)

        # Separate features and labels for training and test data
        X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)

        # Initialize and train the Multinomial Naive Bayes model
        clf = MultinomialNB()
        clf.fit(X_train, y_train)

        # Predict the labels for the test set
        y_test_pred = clf.predict(X_test)

        # Predict probabilities for ROC AUC
        y_test_prob = clf.predict_proba(X_test)[:, 1]

        # Calculate and store the performance metrics
        calculate_performance_metrics(clf, X_test, y_test, y_test_pred, performance_metrics, dim)

        # Generate and plot the ROC curve using probabilities
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        roc_auc = roc_auc_score(y_test, y_test_prob)

        compute_plot_roc_curve(fpr, tpr, f'Dictionary Dimension = {dim} (AUC = {roc_auc:.3f})')

        # Store the classification report for later aggregation
        reports[dim] = classification_report(y_test, y_test_pred, output_dict=True)

    # save the plot and show it
    save_path = 'resources/naive_bayes_roc_curve.png'
    save_plot(save_path)

    # Print the performance metrics
    result_df = aggregate_classification_reports(reports)
    display_classification_reports(result_df)

if __name__ == '__main__':
    run_naive_bayes()