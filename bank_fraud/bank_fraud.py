"""
This file contains the code developed during the second lesson of the course, where we analyze a dataset of bank fraud.
The dataset contains information about bank clients and their transactions, including whether the transaction was
fraudulent caring the privacy of the clients. The goal is to analyze the data and train a model to predict whether a
transaction is fraudulent or not.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, \
    classification_report
from sklearn.preprocessing import StandardScaler

# function to load the dataset
df = pd.read_csv("creditcard.csv")

# function to print some insights about the dataset
print(df.head())
print(df.isnull().sum().max())
print(df.info())

print(df[df.Class == 1].Amount.describe())

# split the dataset into features and classes
X = df.drop("Class", axis=1)
y = df.Class

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# visualize the data and plot the correlation matrix
xv = X.values
yv = y.values
plt.scatter(xv[yv == 0, 1], xv[yv == 0, 2], label="Class 0", alpha=0.5, linewidth=0.2)
plt.scatter(xv[yv == 1, 1], xv[yv == 1, 2], label="Class 1", alpha=0.5, linewidth=0.2, c='r')
plt.legend()
plt.show()

# train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# predict the classes for the test set
y_test_pred = model.predict(X_test)
m = confusion_matrix(y_test, y_test_pred)
ax = sb.heatmap(m, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])
plt.show()

# evaluate the model using precision, accuracy, recall, and f1 score
precision = precision_score(y_test, y_test_pred)
accuracy = accuracy_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"Precision: {precision}")
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# another way to print the metrics
print(model.score(X_test, y_test))
print(classification_report(y_test, y_test_pred))

# as metrics are not good, we can try to balance the dataset
# first, we need to scale the data
s = StandardScaler()
df["Amount"] = s.fit_transform(df["Amount"].values.reshape(-1, 1))

df = df.sample(frac=1)
num_fraud = len(df[df.Class == 1])
fraud_entries = df[df.Class == 1]
non_fraud_entries = df[df["Class"] == 0][:num_fraud]
sub_df = pd.concat([fraud_entries, non_fraud_entries])
sub_df = sub_df.sample(frac=1, random_state=42)
