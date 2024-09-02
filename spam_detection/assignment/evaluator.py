import matplotlib.pyplot as plt
import pandas as pd

from models.knn_classifier import run_knn
from models.logistic_regression_classifier import train_best_model as run_logistic_regression
from models.logistic_regression_classifier import run_lr_comparison
from models.naive_bayes_classifier import train_best_model as run_naive_bayes
from models.svm_classifier import run_svm
from utils.vectorizator import vectorization
from utils.data_analysis import prepare_data, save_plot

rows = []

PATH = '../../dataset/'

# Vectorize the training and testing the datasets with different models
df_train = vectorization(PATH + 'train-mails', 2000)
df_test = vectorization(PATH + 'test-mails', 2000)

# Separate features and labels for training and test data
X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)

# Run Naive Bayes the plot will not be displayed because 1000 is already the best dimension
nb_recall, nb_fpr, nb_tpr, nb_roc_auc = run_naive_bayes(X_train, y_train, X_test, y_test)

rows.append({
    'Model': 'Naive Bayes',
    'Recall': nb_recall,
    'AUC': nb_roc_auc,
    'Best Parameter': 2000
})

# Run Logistic Regression
c_values = [0.0001, 0.1, 1, 10, 1e4]
lr_recall, lr_fpr, lr_tpr, lr_roc_auc, best_c = run_logistic_regression(X_train, y_train, X_test, y_test, c_values)

# Run Logistic Regression comparing the recall and precision metric between test set and training set
run_lr_comparison(X_train, y_train, X_test, y_test, c_values)

rows.append({
    'Model': 'Logistic Regression',
    'Recall': lr_recall,
    'AUC': lr_roc_auc,
    'Best Parameter': best_c
})

# Run KNN
k_values = [4, 6, 8, 10, 15, 20]
knn_recall, knn_fpr, knn_tpr, knn_roc_auc, best_k = run_knn(X_train, y_train, X_test, y_test, k_values)

rows.append({
    'Model': 'KNN',
    'Recall': knn_recall,
    'AUC': knn_roc_auc,
    'Best Parameter': best_k
})

# Run support vector machine
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_recall, svm_fpr, svm_tpr, svm_roc_auc, best_kernel = run_svm(X_train, y_train, X_test, y_test, kernels)

rows.append({
    'Model': 'SVM',
    'Recall': svm_recall,
    'AUC': svm_roc_auc,
    'Best Kernel': best_kernel
})

# Plot the ROC curves
plt.figure()

plt.plot(nb_fpr, nb_tpr, label=f"Naive Bayes (AUC = {nb_roc_auc:.3f})")
plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (C={best_c}, AUC = {lr_roc_auc:.3f})")
plt.plot(knn_fpr, knn_tpr, label=f"KNN (k={best_k}, AUC = {knn_roc_auc:.3f})")
plt.plot(svm_fpr, svm_tpr, label=f"SVM (AUC = {svm_roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC Curves')
plt.legend(loc="lower right")
save_plot('resources/comparison_roc_curve.png')

# Convert the list of dictionaries into a DataFrame for easier comparison
result_df = pd.DataFrame(rows)
print(f"Metrics of comparison between best paramaters in each model \n{result_df.round(4)}")