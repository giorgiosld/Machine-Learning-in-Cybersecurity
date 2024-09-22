from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
import pandas as pd

class Resampler(BaseEstimator, TransformerMixin):
    """
    A class used to resample the dataset to balance the classes with under-sampling, over-sampling or SMOTE

    Attributes:
        mode: The resampling technique to be used. It can be 'under' or 'over'
    """

    def __init__(self, strategy:'undersample'):
        """
        Initialize the Resampler class with the resampling technique
        :param strategy: The resampling technique to be used. It can be 'under', 'over' or 'smote'
        """
        self.strategy = strategy

    def transform(self, X, y):
        """
        Transform the dataset with the resampling technique
        :param X: The features of the dataset
        :param y: The target variable of the dataset
        :return: The resampled dataset
        """
        # Concatenate X and y for easier manipulation
        df = pd.concat([X, y], axis=1)

        # Split by class
        majority_class = df[y == y.value_counts().idxmax()]
        minority_class = df[y == y.value_counts().idxmin()]

        if self.strategy == 'undersample':
            # Undersample majority class to match minority class size
            majority_undersampled = resample(majority_class,
                                             replace=False,  # Do not allow repetition
                                             n_samples=len(minority_class),  # Match minority size
                                             random_state=42)
            resampled_df = pd.concat([majority_undersampled, minority_class])

        elif self.strategy == 'oversample':
            # Oversample minority class to match majority class size
            minority_oversampled = resample(minority_class,
                                            replace=True,  # Allow repetition
                                            n_samples=len(majority_class),  # Match majority size
                                            random_state=42)
            resampled_df = pd.concat([majority_class, minority_oversampled])

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Split resampled dataframe into X and y again
        X_resampled = resampled_df.drop(columns=y.name)
        y_resampled = resampled_df[y.name]

        return X_resampled, y_resampled