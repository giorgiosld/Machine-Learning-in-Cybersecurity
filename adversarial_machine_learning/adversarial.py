import tensorflow as tf
from tensorflow.keras.models import load_model
import foolbox as fb
import matplotlib.pyplot as plt
import os
import eagerpy as ep

class AdversarialMNIST:
    def __init__(self, model_path='mnist_cnn_model.h5'):
        # Load the trained model from train.py
        self.model = load_model(model_path)
        self.fmodel = fb.TensorFlowModel(self.model, bounds=(0, 1))

        # Load and preprocess the MNIST data
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()

    def load_data(self):
        """Load and preprocess MNIST data"""
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Normalize and reshape the dataset
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

        return x_train, y_train, x_test, y_test

    def generate_adversarial_examples(self, attack, x_samples, y_samples, epsilons):
        """Generate adversarial examples"""
        x_samples_ep = ep.astensor(x_samples)
        y_samples_ep = ep.astensor(y_samples)
        adversarial_samples = attack(self.fmodel, x_samples_ep, y_samples_ep, epsilons=epsilons)
        return adversarial_samples.numpy()

    def create_adversarial_grid(self, attack, epsilons, num_samples=10, filename='resources/adversarial_grid.png'):
        """Create a grid of adversarial examples for visualization"""
        # Select 10 samples from the test set
        x_samples = self.x_test[:num_samples]
        y_samples = self.y_test[:num_samples]

        # Generate adversarial examples
        adversarial_samples = self.generate_adversarial_examples(attack, x_samples, y_samples, epsilons)

        # Plot the original and adversarial samples in a grid format
        fig, axes = plt.subplots(num_samples, len(epsilons) + 1, figsize=(15, 20))

        for i in range(num_samples):
            # Original Image
            axes[i, 0].imshow(x_samples[i].squeeze(), cmap="gray")
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')

            # Adversarial Images for each epsilon
            for j, adv_sample in enumerate(adversarial_samples[:, i]):
                axes[i, j + 1].imshow(adv_sample.squeeze(), cmap="gray")
                axes[i, j + 1].set_title(f"Epsilon {epsilons[j]:.2f}")
                axes[i, j + 1].axis('off')

        plt.tight_layout()
        save_dir = os.path.dirname(filename)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(filename)
        plt.show()
        print(f"Adversarial grid saved as {filename}")

    def run_attack(self, attack_type='L2'):
        """Run specified attack and visualize the results"""
        if attack_type == 'L0':
            # L0 variant of the Fast Minimum Norm attack
            attack = fb.attacks.L0FMNAttack(steps=50)  # L0 Fast Minimum Norm Attack
        elif attack_type == 'L1':
            # L1 variant of the Fast Minimum Norm attack
            attack = fb.attacks.L1FMNAttack(steps=50)  # L1 Fast Minimum Norm Attack
        elif attack_type == 'L2':
            # L2 variant of the Fast Minimum Norm attack
            attack = fb.attacks.L2FMNAttack(steps=50)  # L2 Fast Minimum Norm Attack
        elif attack_type == 'Linf':
            # Linf variant of the Fast Minimum Norm attack
            attack = fb.attacks.LinfFMNAttack(steps=50)  # Linf Fast Minimum Norm Attack
        else:
            raise ValueError("Unsupported attack type. Choose from 'L0', 'L1', 'L2', 'Linf'.")

        # Define a range of epsilon values (perturbation levels) to use in the attack
        initial_epsilon = 0.1
        epsilons = [initial_epsilon * (0.9 ** i) for i in range(6)]  # Decreasing epsilons for visualization

        # Create a grid of adversarial examples
        self.create_adversarial_grid(attack, epsilons=epsilons, num_samples=10, filename=f'resources/adversarial_grid_{attack_type}.png')


# Main execution
if __name__ == "__main__":
    adversarial_mnist = AdversarialMNIST(model_path='mnist_cnn_model.h5')

    # Run attack and generate grids for L0, L1, L2, and Linf norms
    norms = ['L0', 'L1', 'L2', 'Linf']
    for norm in norms:
        adversarial_mnist.run_attack(attack_type=norm)
