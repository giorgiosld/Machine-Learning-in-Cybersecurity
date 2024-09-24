from model_trainer import ModelTrainer
from preprocess import Preprocess
from resampler import Resampler


path = 'MachineLearningCVE'

preprocessor = Preprocess(path, 0.6)
d_train, d_test = preprocessor.get_dataset()

X_train, y_train = d_train.drop(columns='Encoded_Label'), d_train['Encoded_Label']
X_test, y_test = d_test.drop(columns='Encoded_Label'), d_test['Encoded_Label']

resampler = Resampler()

X_train_sampled, y_train_sampled = resampler.balanced_undersample(X_train, y_train)
X_test_sampled, y_test_sampled = resampler.balanced_undersample(X_test, y_test)

clf = ModelTrainer('random_forest')

# clf.fit(X_train_sampled, y_train_sampled)
# clf.evaluate(X_test_sampled, y_test_sampled)
clf.fit(X_train, y_train)
clf.evaluate(X_test, y_test)
