import tensorflow as tf
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def prepare_data():
    """Load and preprocess MNIST test data"""
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0
    return x_test, y_test


def create_foolbox_model(keras_model):
    """Convert Keras model to Foolbox model"""
    preprocessing = dict(mean=[0], std=[1])
    bounds = (0, 1)
    fmodel = fb.TensorFlowModel(keras_model, bounds=bounds, preprocessing=preprocessing)
    return fmodel


def generate_adversarial_grid(model, images, labels, attack_class):
    """Generate adversarial examples for different target classes"""
    n_samples = 10  # number of original digits to attack
    results = np.zeros((n_samples, 10, 28, 28))  # 10x10 grid (original x target)

    # Initialize attack once
    attack = attack_class()

    for i in range(n_samples):
        original_image = images[i:i + 1]
        original_label = labels[i]

        # Generate adversarial examples for each target class
        for target in range(10):
            if target != original_label:
                criterion = fb.criteria.TargetedMisclassification(np.array([target]))

                try:
                    # Run the attack with specific parameters
                    adversarial = attack.run(
                        model,
                        original_image,
                        criterion,
                        steps=100,  # Number of steps for the attack
                        random_start=True  # Random initialization
                    )
                    if adversarial is not None:
                        results[i, target] = adversarial.squeeze()
                    else:
                        results[i, target] = original_image.squeeze()
                except Exception as e:
                    print(f"Attack failed for digit {original_label} to target {target}: {str(e)}")
                    results[i, target] = original_image.squeeze()
            else:
                results[i, target] = original_image.squeeze()

    return results


def plot_adversarial_grid(results, attack_name, save_path):
    """Plot and save the grid of adversarial examples"""
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(results[i, j], cmap='gray')
            axes[i, j].axis('off')

    plt.suptitle(f'Adversarial Examples using {attack_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # Load the trained model
    model = load_model('mnist_cnn_model.h5')

    # Prepare data
    x_test, y_test = prepare_data()

    # Create Foolbox model
    fmodel = create_foolbox_model(model)

    # Define attacks (classes, not instances)
    attacks = {
        'L0': fb.attacks.L0FMNAttack,
        'L1': fb.attacks.L1FMNAttack,
        'L2': fb.attacks.L2FMNAttack,
        'LInf': fb.attacks.LInfFMNAttack  # Note: This is the correct capitalization
    }

    # Generate and save results for each attack
    for attack_name, attack_class in attacks.items():
        print(f"Generating adversarial examples using {attack_name} norm...")

        # Select first 10 test images
        images = x_test[:10]
        labels = y_test[:10]

        # Generate adversarial examples
        results = generate_adversarial_grid(fmodel, images, labels, attack_class)

        # Save results
        plot_adversarial_grid(results, attack_name, f'resources/adversarial_grid_{attack_name}.png')
        print(f"Results saved as adversarial_grid_{attack_name}.png")


if __name__ == "__main__":
    main()