import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os


class Preprocess:
    """
    A class used to preproccess the dataset and split into training and test set

    Attributes:
        path_to_files: The path to the directory containing the CSV files
        splitmode: The mode to split the data. If "floating", the data is split into randomize subset, otherwise use day to split
        label_mapping: A dictionary to map the labels to the corresponding values
        label_encoder: The label encoder to encode the labels
    """

    def __init__(self, path_to_files, splitmode="none"):
        """
        Initialize the Preprocess class with the path to the directory containing the CSV files and the split mode
        :param path_to_files: The directory path containing the CSV files
        :param splitmode: The mode of splitting data. If "floating", the data is split into randomize subset, otherwise use day to split
        """
        self.path_to_files = path_to_files
        self.splitmode = splitmode
        self.label_mapping = {
            'benign': 'Benign',
            'dos': 'DoS',
            'ddos': 'DoS',
            'scan': 'PortScan'
        }
        self.label_encoder = LabelEncoder()

    def aggregate_labels(self, df):
        """
        Aggregate the labels in the dataset. This function converts the labels to lowercase and maps them to the corresponding
        :param df: The dataframe containing the dataset with a column named 'Label'
        :return: The function modifies the dataframe in place
        """
        df.columns = df.columns.str.strip()
        df['Label'] = df['Label'].str.lower().apply(
            lambda x: next((value for key, value in self.label_mapping.items() if key in x), 'Exploit')
        )

    def encode_labels(self, df):
        """
        Encode the labels in the dataset using LabelEncoder
        """
        df['Encoded_Label'] = self.label_encoder.fit_transform(df['Label'])

    def ensure_numeric(self, df):
        """
        Ensures all columns in the dataframe are numeric. If a column contains non-numeric data, it will be converted or dropped.
        """
        # Select columns that are not numeric
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns
        for col in non_numeric_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"Could not convert {col}: {e}")

        # Replace Inf/-Inf with NaN
        df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)

        df.fillna(df.mean(), inplace=True)

        return df

    def get_label_mapping(self):
        """
        Get the mapping of encoded labels to their original string values
        """
        return dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))


    @staticmethod
    def plot_distribution(df):
        """
        Plot the distribution of different values in Label feature
        :param df: The dataframe to be plotted
        :return: The function plot the distribution of the labels in place
        """
        label_counts = df['Label'].value_counts()
        # Plot the distribution of labels
        plt.figure(figsize=(10, 6))
        label_counts.plot(kind='bar')
        plt.title('Distribution of Labels in the Dataset')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def see_insight(df):
        """
        This function provides an insight into the dataset. It prints the first five rows, the columns, the number of null entries,
        :param df: The DataFrame to be analyzed
        :return: The function prints the insight of the dataset in place
        """
        print(f"First five rows of the dataset: \n{df.head()}")
        print(f"Null entries in the dataset: \n{df.isnull().sum().max()} ")
        print(f"Label column: \n {df[df.Label == 'Benign'].Label.describe()}")
        print(f"Values in column Label: \n {df['Label'].value_counts()}")
        print(df.groupby('Label').describe())

    def get_dataset(self):
        """
        Reads and processes all CSV files in the specified directory, applies preprocessing steps like label aggregation,
        and splits the dataset based on the specified splitmode.
        :return: Two Pandas DataFrames containing the training and test data
        """
        list_csv = [os.path.join(self.path_to_files, f) for f in os.listdir(self.path_to_files)]

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

        self.aggregate_labels(d_train)
        self.aggregate_labels(d_test)

        self.encode_labels(d_train)
        self.encode_labels(d_test)

        d_train.drop(columns=['Label'], inplace=True)
        d_test.drop(columns=['Label'], inplace=True)

        self.ensure_numeric(d_train)
        self.ensure_numeric(d_test)

        if isinstance(self.splitmode, float) and 0 < self.splitmode < 1:
            combined_df = pd.concat([d_train, d_test], ignore_index=True)
            d_train, d_test = train_test_split(combined_df, test_size=(1 - self.splitmode), random_state=42,
                                               stratify=combined_df['Encoded_Label'])

        return d_train, d_test


if __name__ == "__main__":
    path_to_files = "MachineLearningCVE/"
    preprocessor = Preprocess(path_to_files, 0.8)
    d_train, d_test = preprocessor.get_dataset()
    print(preprocessor.see_insight(d_train))

