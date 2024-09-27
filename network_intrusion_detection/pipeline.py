from model_trainer import ModelTrainer
from preprocess import Preprocess
from resampler import Resampler

# Path to the directory containing the CSV files
path = 'MachineLearningCVE'

# To use different split mode define for floating a value between 0 and 1 as parameter in Preprocess (e.g. 0.6)
# To use day as split mode define 'none' as parameter in Preprocess
preprocessor = Preprocess(path, 0.6)
d_train, d_test = preprocessor.get_dataset()

X_train, y_train = d_train.drop(columns='Encoded_Label'), d_train['Encoded_Label']
X_test, y_test = d_test.drop(columns='Encoded_Label'), d_test['Encoded_Label']

resampler = Resampler()

X_train_sampled, y_train_sampled = resampler.balanced_undersample(X_train, y_train)
X_test_sampled, y_test_sampled = resampler.balanced_undersample(X_test, y_test)

# To use DT or RF define for DT 'decision_tree' and for RF 'random_forest' as parameter in ModelTrainer
clf = ModelTrainer('random_forest')

clf.fit(X_train, y_train)
clf.evaluate(X_test, y_test)
clf.plot(X_test, y_test)
clf.get_feature_importance(X_train)
