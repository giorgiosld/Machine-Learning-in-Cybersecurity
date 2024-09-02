""""
This model implements the Support Vector Machine (SVM) classifier for spam detection.
It uses the `SVC` class from `sklearn` to train the model and predict the labels for the test set.
It also calculates the accuracy, precision, recall, and F1 score for each C value and plots the ROC curve for each C value.
"""
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, classification_report, roc_auc_score

from utils.vectorizator import vectorization
from utils.data_analysis import prepare_data, save_plot, compute_plot_roc_curve, aggregate_classification_reports, \
    display_classification_reports

# Initialize parameters used for the experiment
performance_metrics = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
reports = {}


def run_svm(X_train, y_train, X_test, y_test, kernels):
    best_recall = 0
    best_kernel = None
    best_fpr, best_tpr = None, None
    best_roc_auc = None

    print(f"Running Support Vector Machine with different kernels: {kernels}")

    for kernel in kernels:

        # Initialize and train the SVM model
        svm_clf = SVC(kernel=kernel, probability=True)
        svm_clf.fit(X_train, y_train)

        # Predict probabilities for ROC AUC
        y_test_prob = svm_clf.predict_proba(X_test)[:, 1]

        # Predict the labels for the test set
        y_test_pred = svm_clf.predict(X_test)

        # Generate and plot the ROC curve using probabilities
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        roc_auc = roc_auc_score(y_test, y_test_prob)

        compute_plot_roc_curve(fpr, tpr, f'SVM (kernel={kernel}, AUC = {roc_auc:.3f})')

        # Calculate metrics
        report = classification_report(y_test, y_test_pred, output_dict=True)
        recall = report['1']['recall']
        if recall > best_recall:
            best_recall = recall
            best_kernel = kernel
            best_fpr = fpr
            best_tpr = tpr
            best_roc_auc = roc_auc

        reports[kernel] = report

    save_path = 'resources/svm_roc_curve.png'
    save_plot(save_path)

    # Print the performance metrics
    result_df = aggregate_classification_reports(reports)
    display_classification_reports(result_df)

    return best_recall, best_fpr, best_tpr, best_roc_auc, best_kernel


if __name__ == '__main__':
    PATH = '../../dataset/'

    # Vectorize the training and testing the datasets with different models
    df_train = vectorization(PATH + 'train-mails', 2000)
    df_test = vectorization(PATH + 'test-mails', 2000)

    # Prepare the data for training and testing
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)
    run_svm(X_train, y_train, X_test, y_test, kernels)
