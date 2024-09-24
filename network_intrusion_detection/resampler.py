import pandas as pd

class Resampler():
    """
    A class used to resample the dataset to balance the classes with under-sampling, over-sampling or SMOTE

    Attributes:
        mode: The resampling technique to be used. It can be 'under' or 'over'
    """

    def __init__(self):
        """
        Initialize the Resampler class with the resampling technique
        """

    def balanced_undersample(self, X, y):
        """
        Perform balanced undersampling: undersample all classes to the size of the smallest class.
        :param X: The features of the dataset
        :param y: The target variable of the dataset
        :return: The resampled dataset
        """
        # Concatenate X and y for easier manipulation
        df = pd.concat([X, y], axis=1)

        # Get the size of the smallest class
        min_class_size = y.value_counts().min()

        # Resample each class to the size of the smallest class
        resampled_dfs = []
        for label in y.unique():
            class_df = df[df[y.name] == label]
            resampled_class_df = class_df.sample(n=min_class_size, random_state=42, replace=False)
            resampled_dfs.append(resampled_class_df)

        resampled_df = pd.concat(resampled_dfs)

        # Shuffle the dataset
        resampled_df = resampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split back into X and y
        X_resampled = resampled_df.drop(columns=y.name)
        y_resampled = resampled_df[y.name]

        return X_resampled, y_resampled