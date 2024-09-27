from model_trainer import ModelTrainer
from preprocess import Preprocess

# Path to the directory containing the CSV files
path = 'MachineLearningCVE'

# To use different split mode define for floating a value between 0 and 1 as parameter in Preprocess (e.g. 0.6)
# To use day as split mode define 'none' as parameter in Preprocess
preprocessor = Preprocess(path, 0.6)
d_train, d_test = preprocessor.get_dataset()

X_train, y_train = d_train.drop(columns='Encoded_Label'), d_train['Encoded_Label']
X_test, y_test = d_test.drop(columns='Encoded_Label'), d_test['Encoded_Label']

# To use DT or RF define for DT 'decision_tree' and for RF 'random_forest' as parameter in ModelTrainer
clf = ModelTrainer('decision_tree')

clf.fit(X_train, y_train)
y_pred = clf.evaluate(X_test, y_test)
clf.plot(y_test, y_pred)
clf.get_feature_importance(X_train)
