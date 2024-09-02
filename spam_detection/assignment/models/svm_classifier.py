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
reports = {}

def run_svm(X_train, y_train, X_test, y_test):

    print("Running Support Vector Machine with linear kernel")

    # Initialize and train the SVM model
    svm_clf = SVC(kernel='linear', probability=True)
    svm_clf.fit(X_train, y_train)

    # Predict probabilities for ROC AUC
    y_test_prob = svm_clf.predict_proba(X_test)[:, 1]

    # Predict the labels for the test set
    y_test_pred = svm_clf.predict(X_test)

    # Calculate metrics
    report = classification_report(y_test, y_test_pred, output_dict=True)

    # Generate and plot the ROC curve using probabilities
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = roc_auc_score(y_test, y_test_prob)

    compute_plot_roc_curve(fpr, tpr, f'SVM (AUC = {roc_auc:.3f})')

    reports['Linear'] = report

    save_path = 'resources/svm_roc_curve.png'
    save_plot(save_path)

    # Print the performance metrics
    result_df = aggregate_classification_reports(reports)
    display_classification_reports(result_df)

    return report['1']['recall'], fpr, tpr, roc_auc

if __name__ == '__main__':
    # Load the training and testing datasets
    df_train = vectorization('../../dataset/train-mails', 2000)
    df_test = vectorization('../../dataset/test-mails', 2000)

    # Prepare the data for training and testing
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)
    run_svm(X_train, y_train, X_test, y_test)
