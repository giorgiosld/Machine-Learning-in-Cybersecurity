import os

from sklearn.metrics import precision_score, recall_score, f1_score
from matplotlib import pyplot as plt

def prepare_data(df_train, df_test):
    X_train = df_train.drop(columns=['is_spam'])
    y_train = df_train['is_spam']
    X_test = df_test.drop(columns=['is_spam'])
    y_test = df_test['is_spam']
    return X_train, y_train, X_test, y_test

def calculate_performance_metrics(clf, X_test, y_test, y_test_pred, performance_metrics, k):
    accuracy = clf.score(X_test, y_test)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    performance_metrics.append({
        # 'K': k,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

def compute_plot_roc_curve(fpr, tpr, roc_auc, label):
    plt.plot(fpr, tpr, lw=2, label=f'{label}')
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

def save_plot(path, plt):
    save_dir = os.path.dirname(path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(path)