from sklearn.pipeline import Pipeline

from model_trainer import ModelTrainer
from preprocess import Preprocess
from resampler import Resampler


def build_pipeline(path_to_files, splitmode, resample_strategy, model):
    """
    Build the pipeline for the network intrusion detection system
    :param path_to_files: The path to the directory containing the CSV files
    :param splitmode: The mode to split the data. If "floating", the data is split into randomize subset, otherwise use day to split
    :param resample_strategy: The resampling technique to be used. It can be 'under', 'over' or 'smote'
    :param model: The model to be trained. It can be 'decision_tree' or 'random_forest'
    :return: The pipeline for the network intrusion detection system
    """
    preprocess = Preprocess(path_to_files, splitmode)
    resampler = Resampler(resample_strategy)
    model_trainer = ModelTrainer(model)

    return Pipeline([
        ('preprocess', preprocess),
        ('resampler', resampler),
        ('model_trainer', model_trainer)
    ])

path = 'MachineLearningCVE'

preprocessor = Preprocess(path, 0.6)
d_train, d_test = preprocessor.get_dataset()

X_train, y_train = d_train.drop(columns='Encoded_Label'), d_train['Encoded_Label']
X_test, y_test = d_test.drop(columns='Encoded_Label'), d_test['Encoded_Label']
# X_train, y_train = d_train['Encoded_Label']
# X_test, y_test = d_test['Encoded_Label']

resampler = Resampler("undersample")

X_train_resampled, y_train_resampled = resampler.transform(X_train, y_train)
X_test_resampled, y_test_resampled = resampler.transform(X_test, y_test)

clf = ModelTrainer('decision_tree')
# clf.fit(X_train_resampled, y_train_resampled)
# clf.evaluate(X_test_resampled, y_test_resampled)
clf.fit(X_train, y_train)
clf.evaluate(X_test, y_test)

# pipeline = build_pipeline(path, 'none', 'undersample', 'decision_tree')
# pipeline.fit(X_train, y_train)

# classifier = pipeline.named_steps['model_trainer']
# classifier.evaluate(X_test, y_test)
