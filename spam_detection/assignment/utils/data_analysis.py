import os

import pandas as pd
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

def compute_plot_roc_curve(fpr, tpr, label):
    plt.plot(fpr, tpr, lw=2, label=f'{label}')
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

def save_plot(path):
    save_dir = os.path.dirname(path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(path)
    plt.show()

# Function to aggregate classification reports
def aggregate_classification_reports(reports):
    # Initialize an empty list to collect DataFrames
    dfs = []

    for dim, report in reports.items():
        # Convert the classification report dictionary to a DataFrame
        df = pd.DataFrame(report).transpose()
        # Add a column to indicate the dimension/iteration
        df['dimension'] = dim
        dfs.append(df)

    # Concatenate all DataFrames into one
    result_df = pd.concat(dfs)
    return result_df

# Function to display the DataFrame in a formatted way
def display_classification_reports(result_df):
    # Reset index to have dimension and labels as columns
    result_df = result_df.reset_index()
    # Re-order the columns to have 'dimension' as the first column
    result_df = result_df[['dimension', 'index', 'precision', 'recall', 'f1-score', 'support']]
    # Rename 'index' to 'label'
    result_df = result_df.rename(columns={'index': 'label'})

    # Initialize a string to collect the formatted report
    formatted_report = " dimension        label         precision       recall       f1-score       support\n"

    # Iterate through the DataFrame row by row
    for i, row in result_df.iterrows():
        # If the label is one of the summary statistics, don't include the dimension
        if row['label'] in ['accuracy', 'macro avg', 'weighted avg']:
            formatted_report += f"           {row['label']:>12}         {row['precision']:.3f}           {row['recall']:.3f}        {row['f1-score']:.3f}          {row['support']:6.3f}\n"
        else:
            formatted_report += f"{int(row['dimension']):>10} {row['label']:>12}         {row['precision']:.3f}           {row['recall']:.3f}        {row['f1-score']:.3f}          {row['support']:6.3f}\n"

        # Add a blank line after each dimension's section
        if (i < len(result_df) - 1 and result_df.loc[i, 'dimension'] != result_df.loc[i + 1, 'dimension']):
            formatted_report += "\n"

    # Print the formatted report
    print(formatted_report)