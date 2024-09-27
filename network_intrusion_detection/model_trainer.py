from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import os

class ModelTrainer:
    """
    A class used to train and evaluate the model

    Attributes:
        model: The model to be trained
    """

    def __init__(self, model='decision_tree'):
        """
        Initialize the ModelTrainer class with the model to be trained
        :param model: The model to be trained. It can be 'decision_tree' or 'random_forest'
        """
        if model == 'decision_tree':
            self.model = DecisionTreeClassifier()
        elif model == 'random_forest':
            self.model = RandomForestClassifier()
        else:
            raise ValueError(f"Unknown model: {model}")
        self.labels = ['Benign', 'Dos', 'PortScan', 'Exploit']

    def fit(self, X_train, y_train):
        """
        Fit the model with the training data
        :param X_train: The features of the training data
        :param y_train: The target variable of the training data
        :return: The trained model
        """
        self.model.fit(X_train, y_train)
        return self

    def _save_plot(self, path):
        """
        Private helper function to save the plot to the specified path
        :param path: The file path where the plot will be saved
        """
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(path)
        plt.show()

    def plot(self, y_test, y_pred):
        """
        Function to plot the model and save it to a file
        """
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels, yticklabels=self.labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        path = "resources/cm_random_forest_byFloat.png"
        self._save_plot(path)


    def evaluate(self, X_test, y_test):
        """
        Evaluate the model with the test data
        :param X_test: The features of the test data
        :param y_test: The target variable of the test data
        """
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=self.labels, digits=4))
        return y_pred

    def get_feature_importance(self, X_train):
        """
        Get and plot feature importance for the trained model
        :param X_train: The features of the training data
        """
        # Get the feature importances
        feature_importances = self.model.feature_importances_

        # Create a DataFrame for better readability
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Display the top features
        print("Top Features by Importance:")
        print(importance_df.head(10))

        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'].head(10), importance_df['Importance'].head(10))
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.gca().invert_yaxis()

        # Plot the feature importances
        path = "resources/rf_feature_importance_byFloat.png"
        self._save_plot(path)
