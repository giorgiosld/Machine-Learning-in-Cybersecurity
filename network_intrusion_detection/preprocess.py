import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os

# Define a dictionary for the label mapping
label_mapping = {
    'benign': 'Benign',
    'dos': 'DoS',
    'ddos': 'DoS',
    'scan': 'PortScan'
}

"""
Aggregate the labels in the dataset. This function converts the labels to lowercase and maps them to the corresponding
:param df: The dataframe where analyze the code
"""
def aggregate_labels(df):
    df.columns = df.columns.str.strip()
    print(df['Label'].unique())
    df['Label'] = df['Label'].str.lower().apply(
        lambda x: next((value for key, value in label_mapping.items() if key in x), 'Exploit')
    )
    print(df.head())
    print(df['Label'].unique())


"""
One hot encode the Label column getting a new column for each Label value
:param df: The dataframe where analyze the code
"""
def one_hot_encoding(df):
    print(df['Label'].unique())
    df = pd.get_dummies(df, columns=['Label'])
    print(f"Columns after one-hot encoding: \n {df.head()}")

"""
Plot the distribution of different values in Label feature
:param df: The dataframe to be analyzed
"""
def plot_distribution(df):
    label_counts = df['Label'].value_counts()
    # Plot the distribution of labels
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='bar')
    plt.title('Distribution of Labels in the Dataset')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

"""
This function provides an insight into the dataset. It prints the first five rows, the columns, the number of null entries,
the data type of each column, and the distribution of each label.
:param df: The DataFrame to be analyzed
"""
def see_insight(df):
    print(f"{df.head()}")
    # Since the columns have spaces, we need to remove them
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' ', '_')
    print(f"Features column in the dataset: {df.columns}")
    print(f"Null entries in the dataset: {df.isnull().sum().max()} ")

"""
This function read all CSV files, apply the preprocessing steps, split processed data in two Pandas DataFrames, and
return the latter containing the labeled data. 
:param path_to_files: Path to the directory containing the CSV files
:param splitmode: The mode to split the data. If "floating", the data is split into randomize subset. Otherwise, the data
is splitted from Monday–Wednesday forms the training dataset, and all the data from Thursday/Friday forms the test dataset
"""
def get_dataset(path_to_files, splitmode = "none"):

    list_csv = [os.path.join(path_to_files, f) for f in os.listdir(path_to_files)]

    # Create a single DataFrame with all the data
    # df = pd.concat([pd.read_csv(path_to_files + f) for f in list_csv if f.endswith('.csv')])

    df_train_list = []
    df_test_list = []

    for csv in list_csv:
        filename = os.path.basename(csv)
        if any(day in filename for day in ['Monday', 'Tuesday', 'Wednesday']):
            df_train_list.append(pd.read_csv(csv))
        elif any(day in filename for day in ['Thursday', 'Friday']):
            df_test_list.append(pd.read_csv(csv))

    d_train = pd.concat(df_train_list)
    d_test = pd.concat(df_test_list)

    aggregate_labels(d_train)
    aggregate_labels(d_test)

    if isinstance(splitmode, float) and 0 < splitmode < 1:
        combined_df = pd.concat([d_train, d_test], ignore_index=True)
        d_train, d_test = train_test_split(combined_df, test_size=(1 - splitmode), random_state=42,
                                           stratify=combined_df['Label'])

    plot_distribution(d_train)
    plot_distribution(d_test)

    return d_train, d_test

if __name__ == "__main__":
    path_to_files = "MachineLearningCVE/"
    get_dataset(path_to_files)
