from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import seaborn as sb

from vectorizator import vectorization

df_train, vectorizer = vectorization('train-mails')
df_test, _ = vectorization('test-mails')

X_train = df_train.drop(columns=['is_spam'])
y_train = df_train['is_spam']

X_test = df_test.drop(columns=['is_spam'])
y_test = df_test['is_spam']


clf = MultinomialNB()
clf.fit(X_train, y_train)

y_test_pred = clf.predict(X_test)
m = confusion_matrix(y_test, y_test_pred)
ax = sb.heatmap(m, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])
plt.show()

accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

