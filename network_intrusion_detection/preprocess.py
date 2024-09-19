import pandas as pd
import matplotlib.pyplot as plt

import os

# TODO: fix the umbalnced data because the Label is classified only by BENIGN and the rest are classified as OTHERS
def visualize_dataframe(df):
    # Convert 'Label' to numerical values (BENIGN -> 0, others -> 1)
    df['Label_numeric'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    # Extract feature values  and labels
    xv = df[['Flow_Duration', 'Total_Fwd_Packets']].values
    yv = df['Label_numeric'].values
    plt.figure(figsize=(10, 6))
    plt.scatter(xv[yv == 0, 0], xv[yv == 0, 1], label="BENIGN", alpha=0.5, linewidth=0.2)
    plt.scatter(xv[yv == 1, 0], xv[yv == 1, 1], label="Others", alpha=0.5, linewidth=0.2, color='r')
    plt.xlabel('Flow Duration')
    plt.ylabel('Total Fwd Packets')
    plt.title('Flow Duration vs Total Fwd Packets')
    plt.legend()
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
    print(f"Query the dataset: {df.info()} ")
    # Filter data by each label
    print(f"Label distribution: {df[df.Label == 'BENIGN'].Destination_Port.describe()}")

"""
This function read all CSV files, apply the preprocessing steps, split processed data in two Pandas DataFrames, and
return the latter containing the labeled data. 
:param path_to_files: Path to the directory containing the CSV files
:param splitmode: The mode to split the data. If "floating", the data is split into randomize subset. Otherwise, the data
is splitted from Monday–Wednesday forms the training dataset, and all the data from Thursday/Friday forms the test dataset
"""
def get_dataset(path_to_files, splitmode):
    list_csv = os.listdir(path_to_files)
    # Create a single DataFrame with all the data
    df = pd.concat([pd.read_csv(path_to_files + f) for f in list_csv if f.endswith('.csv')])
    see_insight(df)
    visualize_dataframe(df)
    d_training, d_testing = None, None
    return d_training, d_testing

if __name__ == "__main__":
    path_to_files = "MachineLearningCVE/"
    get_dataset(path_to_files, "split")