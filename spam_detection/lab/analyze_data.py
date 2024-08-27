"""
This script analyzes data by training a Multinomial Naive Bayes model using vectorized data from `vectorizator.py` and
evaluates the model's performance through various metrics, including Confusion Matrix, Accuracy, Precision, Recall, and
F1 Score.
"""
from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sb

from vectorizator import vectorization

# Vectorize the training and testing datasets
df_train = vectorization('../dataset/train-mails')
df_test = vectorization('../dataset/test-mails')

# Separate features and labels for training and test data
X_train = df_train.drop(columns=['is_spam'])
y_train = df_train['is_spam']

X_test = df_test.drop(columns=['is_spam'])
y_test = df_test['is_spam']

# Initialize and train the Multinomial Naive Bayes model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict the labels for the test set
y_test_pred = clf.predict(X_test)

# Generate and plot the confusion matrix
m = confusion_matrix(y_test, y_test_pred)
ax = sb.heatmap(m, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])
plt.show()

# Generate and plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Calculate and print Accuracy, Precision, Recall, and F1 Score
accuracy = clf.score(X_test, y_test)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")