import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os


class MNISTModel:
    def __init__(self, epochs=5, batch_size=64, num_classes=10, input_shape=(28, 28, 1)):
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self.build_model()
        self.history = None

    def load_and_preprocess_data(self):
        """Load and preprocess MNIST data"""
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Check for NaN values
        if np.isnan(x_train).any() or np.isnan(x_test).any():
            raise ValueError("NaN values found in the dataset")

        # Normalize and reshape the dataset
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

        # Label encoding the target variable
        y_train = tf.one_hot(y_train.astype(np.int32), depth=self.num_classes)
        y_test = tf.one_hot(y_test.astype(np.int32), depth=self.num_classes)

        return x_train, y_train, x_test, y_test

    def build_model(self):
        """Build the convolutional neural network model"""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=self.input_shape),
            tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy',
                      metrics=['acc'])
        return model

    def train_model(self, x_train, y_train):
        """Train the model"""
        self.history = self.model.fit(x_train, y_train,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      validation_split=0.1)

    def evaluate_model(self, x_test, y_test):
        """Evaluate the model on the test set"""
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        print(f"Test accuracy: {test_acc}")
        return test_loss, test_acc

    def save_model(self, filepath='mnist_cnn_model.h5'):
        """Save the trained model to a file"""
        self.model.save(filepath)
        print(f"Model saved as {filepath}")

    def plot_results(self):
        """Plot training history"""
        if self.history is None:
            raise ValueError("No training history found. Please train the model first.")

        fig, ax = plt.subplots(2, 1)

        # Plot training and validation loss
        ax[0].plot(self.history.history['loss'], color='b', label="Training Loss")
        ax[0].plot(self.history.history['val_loss'], color='r', label="Validation Loss")
        ax[0].legend(loc='best', shadow=True)

        # Plot training and validation accuracy
        ax[1].plot(self.history.history['acc'], color='b', label="Training Accuracy")
        ax[1].plot(self.history.history['val_acc'], color='r', label="Validation Accuracy")
        ax[1].legend(loc='best', shadow=True)

        plt.tight_layout()
        self._save_plot('resources/loss_accuracy_plot.png')

    def create_confusion_matrix(self, x_test, y_test):
        """Create confusion matrix"""
        Y_pred = self.model.predict(x_test)
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        Y_true = np.argmax(y_test, axis=1)

        confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes)

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, annot=True, fmt='g')
        self._save_plot('resources/confusion_matrix.png')

    @staticmethod
    def _save_plot(path):
        """Helper function to save the plot to the specified path"""
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(path)
        plt.show()


# Main execution
if __name__ == "__main__":
    mnist_model = MNISTModel(epochs=5, batch_size=64)

    # Load and preprocess data
    x_train, y_train, x_test, y_test = mnist_model.load_and_preprocess_data()

    # Train the model
    mnist_model.train_model(x_train, y_train)

    # Save the model to file
    mnist_model.save_model('mnist_cnn_model.h5')

    # Evaluate the model
    mnist_model.evaluate_model(x_test, y_test)

    # Plot the results
    mnist_model.plot_results()

    # Create confusion matrix
    mnist_model.create_confusion_matrix(x_test, y_test)
