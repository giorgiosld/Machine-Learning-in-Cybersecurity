"""
This script implements the K-Nearest Neighbors classifier for spam detection.
It uses the `KNeighborsClassifier` class from `sklearn` to train the model and predict the labels for the test set.
It also calculates the accuracy, precision, recall, and F1 score for each K value and plots the ROC curve for each K value.
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, classification_report, roc_auc_score

from utils.vectorizator import vectorization
from utils.data_analysis import prepare_data, calculate_performance_metrics, compute_plot_roc_curve, save_plot, \
    display_classification_reports, aggregate_classification_reports

# Initialize parameters used for the experiment
performance_metrics = []
k_params = [4, 6, 8, 10, 15, 20]
reports = {}

def run_knn(X_train, y_train, X_test, y_test, k_params):

    best_recall = 0
    best_k = None
    best_fpr, best_tpr = None, None
    best_roc_auc = None

    print(f"Running K-Nearest Neighbors with k values: {k_params}")
    for k in k_params:

        # Initialize and train the K-Nearest Neighbors model
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train, y_train)

        # Predict probabilities for ROC AUC
        y_test_prob = neigh.predict_proba(X_test)[:, 1]

        # Predict the labels for the test set
        y_test_pred = neigh.predict(X_test)

        # Generate and plot the ROC curve using probabilities
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        roc_auc = roc_auc_score(y_test, y_test_prob)

        compute_plot_roc_curve(fpr, tpr, f'KNN (k={k}, AUC = {roc_auc:.3f})')

        report = classification_report(y_test, y_test_pred, output_dict=True)
        recall = report['1']['recall']
        if recall > best_recall:
            best_recall = recall
            best_k = k
            best_fpr = fpr
            best_tpr = tpr
            best_roc_auc = roc_auc

        reports[k] = report

    # save the plot and show it
    save_path = 'resources/knn_roc_curve.png'
    save_plot(save_path)

    # Print the performance metrics
    result_df = aggregate_classification_reports(reports)
    display_classification_reports(result_df)

    return best_recall, best_fpr, best_tpr, best_roc_auc, best_k


if __name__ == '__main__':
    # Vectorize the training and testing datasets
    df_train = vectorization('../../dataset/train-mails', 2000)
    df_test = vectorization('../../dataset/test-mails', 2000)

    # Separate features and labels for training and test data
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)
    run_knn(X_train, y_train, X_test, y_test, k_params)