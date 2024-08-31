import matplotlib.pyplot as plt

from models.knn_classifier import train_best_model as run_knn
from models.logistic_regression_classifier import train_best_model as run_logistic_regression
from models.naive_bayes_classifier import train_best_model as run_naive_bayes
from utils.vectorizator import vectorization
from utils.data_analysis import prepare_data, save_plot

# TODO: Implement the possibility to pass the path for the dataset

# Vectorize the training and testing the datasets with different models
df_train = vectorization('../../dataset/train-mails', 1000)
df_test = vectorization('../../dataset/test-mails', 1000)

# Separate features and labels for training and test data
X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)

# Run Naive Bayes
nb_recall, nb_fpr, nb_tpr, nb_roc_auc = run_naive_bayes(X_train, y_train, X_test, y_test)

# Run Logistic Regression
c_values = [0.0001, 0.1, 1, 10, 1e4]
lr_recall, lr_fpr, lr_tpr, lr_roc_auc, best_c = run_logistic_regression(X_train, y_train, X_test, y_test, c_values)

# Run KNN
k_values = [4, 6, 8, 10, 15, 20]
knn_recall, knn_fpr, knn_tpr, knn_roc_auc, best_k = run_knn(X_train, y_train, X_test, y_test, k_values)

# Plot the ROC curves
plt.figure()

plt.plot(nb_fpr, nb_tpr, label=f"Naive Bayes (AUC = {nb_roc_auc:.3f})")
plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (C={best_c}, AUC = {lr_roc_auc:.3f})")
plt.plot(knn_fpr, knn_tpr, label=f"KNN (k={best_k}, AUC = {knn_roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC Curves')
plt.legend(loc="lower right")
save_plot('resources/comparison_roc_curve.png')
