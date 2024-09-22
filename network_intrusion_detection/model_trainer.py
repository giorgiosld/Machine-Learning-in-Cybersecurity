from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ModelTrainer(BaseEstimator):
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

    def fit(self, X_train, y_train):
        """
        Fit the model with the training data
        :param X_train: The features of the training data
        :param y_train: The target variable of the training data
        :return: The trained model
        """
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model with the test data
        :param X_test: The features of the test data
        :param y_test: The target variable of the test data
        """
        labels = ['Benign', 'Dos', 'PortScan', 'Exploit']
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=labels))

        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show() 

