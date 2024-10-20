import tensorflow as tf
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import eagerpy as ep


def prepare_data():
    """Load and preprocess MNIST test data, ensuring digits 0-9 are in order"""
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

    ordered_images = []
    ordered_labels = []

    for digit in range(10):
        digit_idx = np.where(y_test == digit)[0][0]
        ordered_images.append(x_test[digit_idx])
        ordered_labels.append(y_test[digit_idx])

    return np.array(ordered_images), np.array(ordered_labels)


def create_foolbox_model(keras_model):
    """Convert Keras model to Foolbox model"""
    preprocessing = dict(mean=0.0, std=1.0)
    bounds = (0, 1)
    fmodel = fb.TensorFlowModel(keras_model, bounds=bounds, preprocessing=preprocessing)
    return fmodel


def generate_adversarial_grid(fmodel, images, labels, attack_class):
    """Generate adversarial examples using specified Foolbox attack"""
    n_samples = 10
    results = np.zeros((n_samples, 10, 28, 28))

    # Base attack parameters
    attack_params = {
        'steps': 200,
        'binary_search_steps': 10
    }

    epsilon_sets = {
        'L2': [0.1, 0.5, 1.0, 1.5, 2.0],
        'L1': [1.0, 3.0, 5.0, 7.0, 10.0],
        'L0': [1, 5, 10, 15, 20],
        'LInf': [0.01, 0.05, 0.1, 0.2, 0.3]
    }

    # Get attack name from class
    attack_name = attack_class.__name__
    attack_type = next((k for k in epsilon_sets.keys() if k in attack_name), 'L2')
    epsilons = epsilon_sets[attack_type]

    # Initialize attack
    attack = attack_class(**attack_params)

    for i in range(n_samples):
        original_image = images[i:i + 1]
        original_label = labels[i]

        # Store original image in diagonal position
        results[i, original_label] = original_image.squeeze()

        for target in range(10):
            if target != original_label:
                success = False

                # Convert inputs to EagerPy tensors
                images_ep = ep.astensor(original_image)
                target_ep = ep.astensor(np.array([target]))
                criterion = fb.criteria.TargetedMisclassification(target_ep)

                # Try different epsilon values
                for epsilon in epsilons:
                    if success:
                        break

                    try:
                        print(
                            f"\nTrying {attack_type} attack from {original_label} to {target} with epsilon: {epsilon}")

                        # Run attack with current epsilon
                        raw_advs, clipped_advs, is_successful = attack(fmodel, images_ep, criterion, epsilons=[epsilon])

                        if is_successful.numpy().any():
                            adversarial = clipped_advs.numpy()
                            results[i, target] = adversarial.squeeze()
                            success = True

                            # Calculate and print perturbation magnitude
                            l2_dist = np.linalg.norm((adversarial - original_image).reshape(-1))
                            linf_dist = np.max(np.abs(adversarial - original_image))
                            print(f"Success: {original_label} -> {target}")
                            print(f"L2 distance: {l2_dist:.4f}")
                            print(f"L∞ distance: {linf_dist:.4f}")
                            break

                    except Exception as e:
                        print(f"Attack failed with epsilon {epsilon}: {str(e)}")
                        continue

                if not success:
                    print(f"\nAll attempts failed for {original_label} -> {target}")
                    results[i, target] = original_image.squeeze()

    return results


def plot_adversarial_grid(results, attack_name, filename):
    """Plot and save grid of adversarial examples"""
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    plt.suptitle(f'Adversarial Examples using {attack_name}', fontsize=16)

    # Add column headers
    for j in range(10):
        axes[0, j].set_title(f'Target: {j}', fontsize=10)

    # Add row labels
    for i in range(10):
        axes[i, 0].set_ylabel(f'Original: {i}', rotation=0, labelpad=25, fontsize=10)

    # Plot images
    for i in range(10):
        for j in range(10):
            im = axes[i, j].imshow(results[i, j], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')

            # Highlight diagonal (original images) with a border
            if i == j:
                axes[i, j].patch.set_edgecolor('red')
                axes[i, j].patch.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()


def main():
    print("Loading model...")
    model = load_model('mnist_cnn_model.h5')

    print("Preparing data...")
    x_test, y_test = prepare_data()

    print("Creating Foolbox model...")
    fmodel = create_foolbox_model(model)

    # Define attacks
    attacks = {
        'L2': fb.attacks.L2FMNAttack,
        'L1': fb.attacks.L1FMNAttack,
        'L0': fb.attacks.L0FMNAttack,
        'LInf': fb.attacks.LInfFMNAttack
    }

    # Generate and save results for each attack
    for attack_name, attack_class in attacks.items():
        print(f"\nGenerating adversarial examples using {attack_name} norm...")
        results = generate_adversarial_grid(fmodel, x_test, y_test, attack_class)
        plot_adversarial_grid(results, attack_name, f'resources/adversarial_grid_{attack_name}.png')
        print(f"Results saved as adversarial_grid_{attack_name}.png")


if __name__ == "__main__":
    main()