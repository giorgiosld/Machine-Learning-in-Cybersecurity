"""
This script implements the Logistic Regression classifier for spam detection.
It uses the `LogisticRegression` class from `sklearn` to train the model and predict the labels for the test set.
It also calculates the accuracy, precision, recall, and F1 score for each C value and plots the ROC curve for each C value.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, classification_report, roc_auc_score

from utils.vectorizator import vectorization
from utils.data_analysis import prepare_data, save_plot, compute_plot_roc_curve, \
    aggregate_classification_reports, display_classification_reports

performance_metrics = []
c = [0.0001, 0.1, 1, 10, 1e4]
reports = {}

def train_best_model(X_train, y_train, X_test, y_test, c_values):

    best_recall = 0
    best_c = None
    best_fpr, best_tpr = None, None
    best_roc_auc = None

    print(f"Running Logistic Regression with C values: {c_values}")
    for c_val in c_values:
        # Initialize and train the Logistic Regression model
        clf = LogisticRegression(C=c_val)
        clf.fit(X_train, y_train)

        # Predict probabilities for ROC AUC
        y_test_prob = clf.predict_proba(X_test)[:, 1]

        # Predict the labels for the test set
        y_test_pred = clf.predict(X_test)

        # Generate and plot the ROC curve using probabilities
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        roc_auc = roc_auc_score(y_test, y_test_prob)

        compute_plot_roc_curve(fpr, tpr, f'Logistic Regression (C={c_val}, AUC = {roc_auc:.3f})')

        # Calculate performance metrics
        report = classification_report(y_test, y_test_pred, output_dict=True)
        recall = report['1']['recall']
        if recall > best_recall:
            best_recall = recall
            best_c = c_val
            best_fpr = fpr
            best_tpr = tpr
            best_roc_auc = roc_auc

        # Store the classification report for later aggregation
        reports[c_val] = report

    # save the plot and show it
    save_path = 'resources/lr_roc_curve.png'
    save_plot(save_path)

    # Print the performance metrics
    result_df = aggregate_classification_reports(reports)
    display_classification_reports(result_df)

    return best_recall, best_fpr, best_tpr, best_roc_auc, best_c

def run_lr_comparison(X_train, y_train, X_test, y_test, c_values):

    rows = []

    print(f"Running Logistic Regression comparing training and test set with C values: {c_values}")
    for c_val in c:

        # Initialize and train the Logistic Regression model
        clf = LogisticRegression(C=c_val)
        clf.fit(X_train, y_train)

        # Prediction for training set
        y_train_pred = clf.predict(X_train)

        # Predict the labels for the test set
        y_test_pred = clf.predict(X_test)

        # Extract the precision and recall for both sets
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)

        # Append the relevant metrics to the rows list
        rows.append({
            'C_value': c_val,
            'Train_Precision': train_report['weighted avg']['precision'],
            'Train_Recall': train_report['weighted avg']['recall'],
            'Test_Precision': test_report['weighted avg']['precision'],
            'Test_Recall': test_report['weighted avg']['recall']
        })

    # Convert the list of dictionaries into a DataFrame for easier comparison
    result_df = pd.DataFrame(rows)
    print(result_df.round(4))

if __name__ == '__main__':
    PATH = '../../dataset/'

    # Vectorize the training and testing datasets
    df_train = vectorization(PATH + 'train-mails', 2000)
    df_test = vectorization(PATH + 'test-mails', 2000)

    # Separate features and labels for training and test data
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)
    train_best_model(X_train, y_train, X_test, y_test, c)
    run_lr_comparison(X_train, y_train, X_test, y_test, c)