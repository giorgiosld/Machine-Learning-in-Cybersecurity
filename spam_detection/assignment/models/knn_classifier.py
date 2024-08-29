"""
This script implements the K-Nearest Neighbors classifier for spam detection.
It uses the `KNeighborsClassifier` class from `sklearn` to train the model and predict the labels for the test set.
It also calculates the accuracy, precision, recall, and F1 score for each K value and plots the ROC curve for each K value.
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

from utils.vectorizator import vectorization
from utils.data_analysis import prepare_data, calculate_performance_metrics, compute_plot_roc_curve, save_plot

performance_metrics = []
k_params = [4, 6, 8, 10, 15, 20]

for k in k_params:

    # Vectorize the training and testing datasets
    df_train = vectorization('../../dataset/train-mails', 1000)
    df_test = vectorization('../../dataset/test-mails', 1000)

    # Separate features and labels for training and test data
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)

    # Initialize and train the K-Nearest Neighbors model
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)

    # Predict the labels for the test set
    y_test_pred = neigh.predict(X_test)

    # Calculate and store the performance metrics
    calculate_performance_metrics(neigh, X_test, y_test, y_test_pred, performance_metrics, k)

    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_pred)
    roc_auc = auc(fpr, tpr)

    compute_plot_roc_curve(fpr, tpr, roc_auc, f'ROC Curve (AUC = {roc_auc:.2f})')


# Define the path to save the file
save_path = '../resources/knn_roc_curve.png'
save_plot(save_path, plt)

# Show the plot
plt.show()

# Print the performance metrics
performance_metrics_df = pd.DataFrame(performance_metrics, index=k_params)
print(performance_metrics_df)