import tensorflow as tf
import foolbox as fb
import eagerpy as ep
import numpy as np
import matplotlib.pyplot as plt
from foolbox.attacks import L2FMNAttack, L0FMNAttack, L1FMNAttack, LInfFMNAttack


class AdversarialTester:
    def __init__(self, model_path, attack_list, epsilons):
        self.model_path = model_path
        self.attacks = attack_list
        self.epsilons = epsilons
        self.fmodel = None
        self.images = None
        self.labels = None

    def load_model(self):
        """Load the pre-trained MNIST model and wrap it with Foolbox."""
        model = tf.keras.models.load_model(self.model_path)
        preprocessing = dict(mean=0.0, std=1.0)
        self.fmodel = fb.TensorFlowModel(model, bounds=(0, 1), preprocessing=preprocessing)
        print("Model loaded successfully.")

    def prepare_data(self):
        """Prepare MNIST data for testing, ensuring we have one of each digit."""
        mnist = tf.keras.datasets.mnist
        (_, _), (x_test, y_test) = mnist.load_data()

        # Initialize arrays to store one example of each digit (0-9)
        unique_digits = {}
        for i in range(len(y_test)):
            if y_test[i] not in unique_digits:
                unique_digits[y_test[i]] = x_test[i]
            if len(unique_digits) == 10:
                break

        # Prepare selected images and labels
        x_selected = np.array([unique_digits[digit] for digit in range(10)])
        y_selected = np.array([digit for digit in range(10)])

        # Normalize and reshape the dataset
        x_selected = x_selected.reshape(x_selected.shape[0], 28, 28, 1).astype('float32') / 255.0
        y_selected = y_selected.astype(np.int32)

        # Convert data to EagerPy tensors backed by TensorFlow
        self.images = ep.astensor(tf.convert_to_tensor(x_selected))
        self.labels = ep.astensor(tf.convert_to_tensor(y_selected))

        print("Data prepared successfully.")

    def evaluate_clean_accuracy(self):
        """Evaluate the accuracy of the model on clean data."""
        clean_acc = fb.accuracy(self.fmodel, self.images, self.labels)
        print(f"Clean accuracy: {clean_acc * 100:.1f} %")

    def perform_attack(self, attack, attack_name):
        """Perform an adversarial attack."""
        if attack_name == "L0FMNAttack":
            raw_adversarial, clipped_adversarial, success = attack(self.fmodel, self.images, self.labels,
                                                                   epsilons=None)
            num_epsilons = 1
        else:
            raw_adversarial, clipped_adversarial, success = attack(self.fmodel, self.images, self.labels,
                                                                   epsilons=self.epsilons)
            num_epsilons = len(self.epsilons)

        return raw_adversarial, clipped_adversarial, success, num_epsilons

    def analyze_results(self, clipped_adversarial, success, attack_name):
        """Analyze and print the results of an adversarial attack."""
        print(f"\nResults for {attack_name}:")
        if attack_name == "L0FMNAttack":
            acc = fb.accuracy(self.fmodel, clipped_adversarial, self.labels)
            perturbation_sizes = (clipped_adversarial - self.images).norms.l0(axis=(1, 2, 3)).numpy()
            print(f"  Attack: Accuracy after attack: {acc * 100:4.1f} %, Perturbation size: {perturbation_sizes}")
        else:
            for eps, advs_, succ_ in zip(self.epsilons, clipped_adversarial, success):
                acc = fb.accuracy(self.fmodel, advs_, self.labels)
                perturbation_sizes = (advs_ - self.images).norms.l2(axis=(1, 2, 3)).numpy()
                print(
                    f"  Epsilon {eps:<6}: Accuracy after attack: {acc * 100:4.1f} %, Success: {succ_}, Perturbation size: {perturbation_sizes}")

    def visualize_attack(self, clipped_adversarial, success, attack_name, num_epsilons):
        """Visualize the results of an adversarial attack."""
        num_images = len(self.images)

        # Set up the figure to visualize the grid for the current attack
        fig, axes = plt.subplots(num_images, num_epsilons + 1 if attack_name != "L0FMNAttack" else 2, figsize=(15, 15))
        fig.suptitle(f'Adversarial Examples for {attack_name}', fontsize=20)

        for i in range(num_images):
            # Plot the original image in the first column
            ax = axes[i, 0]
            original_img = self.images[i].numpy().squeeze()
            ax.imshow(original_img, cmap='gray')
            if i == 0:
                ax.set_title(f'Original', fontsize=10)
            ax.axis('off')

            # Plot the adversarial images for each epsilon value in subsequent columns
            if attack_name == "L0FMNAttack":
                ax = axes[i, 1]
                adversarial_img = clipped_adversarial[i].numpy().squeeze()
                ax.imshow(adversarial_img, cmap='gray')
                if i == 0:
                    ax.set_title(f'L0 Attack', fontsize=10)
                ax.axis('off')
            else:
                for j in range(num_epsilons):
                    ax = axes[i, j + 1]
                    adversarial_img = clipped_adversarial[j][i].numpy().squeeze()
                    ax.imshow(adversarial_img, cmap='gray')
                    if i == 0:
                        ax.set_title(f'Eps: {self.epsilons[j]}', fontsize=10)
                    ax.axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'resources/adversarial_grid_{attack_name}.png')
        plt.show()

    def run_all_attacks(self):
        """Run all attacks specified and visualize the results."""
        for attack, attack_name in self.attacks:
            raw_adversarial, clipped_adversarial, success, num_epsilons = self.perform_attack(attack, attack_name)
            self.analyze_results(clipped_adversarial, success, attack_name)
            self.visualize_attack(clipped_adversarial, success, attack_name, num_epsilons)
            print(f"Completed visualization for {attack_name}.")


if __name__ == "__main__":
    # List of attacks to apply
    attack_list = [
        (L0FMNAttack(), "L0FMNAttack"),
        (L1FMNAttack(), "L1FMNAttack"),
        (L2FMNAttack(), "L2FMNAttack"),
        (LInfFMNAttack(), "LInfFMNAttack"),
        (fb.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(), "L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack"),
    ]

    # Define epsilon values to use in the attacks
    epsilons = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1, 2, 5, 7, 9]

    # Instantiate and run the AdversarialTester
    tester = AdversarialTester("mnist_cnn_model.h5", attack_list, epsilons)
    tester.load_model()
    tester.prepare_data()
    tester.evaluate_clean_accuracy()
    tester.run_all_attacks()
