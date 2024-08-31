"""
This script implements the Logistic Regression classifier for spam detection.
It uses the `LogisticRegression` class from `sklearn` to train the model and predict the labels for the test set.
It also calculates the accuracy, precision, recall, and F1 score for each C value and plots the ROC curve for each C value.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report, roc_auc_score

from utils.vectorizator import vectorization
from utils.data_analysis import prepare_data, calculate_performance_metrics, save_plot, compute_plot_roc_curve, \
    aggregate_classification_reports, display_classification_reports

performance_metrics = []
c = [0.0001, 0.1, 1, 10, 1e4]
reports = {}

def train_best_model(X_train, y_train, X_test, y_test, c_values):

    best_recall = 0
    best_c = None
    best_fpr, best_tpr = None, None
    best_roc_auc = None

    for c_val in c_values:
        # Initialize and train the Logistic Regression model
        clf = LogisticRegression(C=c_val)
        clf.fit(X_train, y_train)

        # Prediction for training set
        y_train_pred = clf.predict(X_train)
        y_train_prob = clf.predict_proba(X_train)[:, 1]

        # Calculate performance metrics
        report = classification_report(y_train, y_train_pred, output_dict=True)
        recall = report['1']['recall']

        # Calculate the ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_train, y_train_prob)
        roc_auc = roc_auc_score(y_train, y_train_prob)

        if recall > best_recall:
            best_recall = recall
            best_c = c_val
            best_fpr = fpr
            best_tpr = tpr
            best_roc_auc = roc_auc

    return best_recall, best_fpr, best_tpr, best_roc_auc, best_c

def run_logistic_regression():

    for c_val in c:

        # Vectorize the training and testing datasets
        df_train = vectorization('../../dataset/train-mails', 1000)
        df_test = vectorization('../../dataset/test-mails', 1000)

        # Separate features and labels for training and test data
        X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)

        # Initialize and train the Logistic Regression model
        clf = LogisticRegression(C=c_val)
        clf.fit(X_train, y_train)

        # Prediction for training set
        y_train_pred = clf.predict(X_train)
        y_train_prob = clf.predict_proba(X_train)[:, 1]  # Probabilities for the positive class

        # Predict the labels for the test set
        y_test_pred = clf.predict(X_test)
        y_test_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

        # Calculate and store the performance metrics
        calculate_performance_metrics(clf, X_test, y_test, y_test_pred, performance_metrics, 1000)

        # Generate and plot the ROC curve using probabilities
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        roc_auc = roc_auc_score(y_test, y_test_prob)

        compute_plot_roc_curve(fpr, tpr, f'Logistic Regression (C={c_val}, AUC = {roc_auc:.3f})')

        # Store the classification report for later aggregation (for both training and test)
        reports[f'Training_C_{c_val}'] = classification_report(y_train, y_train_pred, output_dict=True)
        reports[f'Test_C_{c_val}'] = classification_report(y_test, y_test_pred, output_dict=True)

        # Store the classification report for later aggregation
        # reports[c_val] = classification_report(y_test, y_test_pred, output_dict=True)

    # save the plot and show it
    save_path = 'resources/lr_roc_curve.png'
    save_plot(save_path)

    # Print the performance metrics
    result_df = aggregate_classification_reports(reports)
    display_classification_reports(result_df)

if __name__ == '__main__':
    run_logistic_regression()