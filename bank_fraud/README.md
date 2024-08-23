# Bank Fraud Detection

This directory contains the code for the bank fraud detection project. The project is a part of the course "Machine Learning in Cybersecurity" at Reykjavik University.

## Objective

The objective of this project is to analyze a dataset of bank transactions and develop a machine learning model to predict whether a transaction is fraudulent or not.

## Setup
+ Extract the `creditcard.csv` file from the `creditcard.zip` archive.
    ```bash
    unzip creditcard.zip
    ```

## Usage

1. Load the dataset:
    ```python
    df = pd.read_csv("creditcard.csv")
    ```

2. Analyze the dataset:
    ```python
    print(df.head())
    print(df.isnull().sum().max())
    print(df.info())
    print(df[df.Class == 1].Amount.describe())
    ```

3. Split the dataset into features and classes:
    ```python
    X = df.drop("Class", axis=1)
    y = df.Class
    ```

4. Split data into training and testing sets:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

5. Train a logistic regression model:
    ```python
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    ```

6. Predict the classes for the test set and evaluate the model:
    ```python
    y_test_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_test_pred))
    print(precision_score(y_test, y_test_pred))
    print(accuracy_score(y_test, y_test_pred))
    print(recall_score(y_test, y_test_pred))
    print(f1_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))
    ```

7. Visualize the data and plot the correlation matrix:
    ```python
    plt.scatter(X.values[y.values == 0, 1], X.values[y.values == 0, 2], label="Class 0", alpha=0.5, linewidth=0.2)
    plt.scatter(X.values[y.values == 1, 1], X.values[y.values == 1, 2], label="Class 1", alpha=0.5, linewidth=0.2, c='r')
    plt.legend()
    plt.show()
    ```