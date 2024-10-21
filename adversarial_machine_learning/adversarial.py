import tensorflow as tf
import foolbox as fb
import eagerpy as ep
import numpy as np
import matplotlib.pyplot as plt
from foolbox.attacks import L2FMNAttack, L0FMNAttack, L1FMNAttack, LInfFMNAttack

# Load your MNIST model
model_path = "mnist_cnn_model.h5"
model = tf.keras.models.load_model(model_path)

# Define preprocessing used by Foolbox
preprocessing = dict(mean=0.0, std=1.0)

# Wrap the model in a Foolbox model
fmodel = fb.TensorFlowModel(model, bounds=(0, 1), preprocessing=preprocessing)

# Load MNIST test data
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()

# Initialize arrays to store one example of each digit (0-9)
unique_digits = {}
for i in range(len(y_test)):
    if y_test[i] not in unique_digits:
        unique_digits[y_test[i]] = x_test[i]
    # Stop once we have one of each digit
    if len(unique_digits) == 10:
        break

# Prepare data for Foolbox attack
x_selected = np.array([unique_digits[digit] for digit in range(10)])
y_selected = np.array([digit for digit in range(10)])

# Normalize and reshape the dataset
x_selected = x_selected.reshape(x_selected.shape[0], 28, 28, 1).astype('float32') / 255.0
y_selected = y_selected.astype(np.int32)

# Convert data to EagerPy tensors backed by TensorFlow
images = ep.astensor(tf.convert_to_tensor(x_selected))
labels = ep.astensor(tf.convert_to_tensor(y_selected))

# Clean accuracy
clean_acc = fb.accuracy(fmodel, images, labels)
print(f"Clean accuracy: {clean_acc * 100:.1f} %")

# List of attacks to apply
attacks = [
    (L0FMNAttack(), "L0FMNAttack"),
    (L1FMNAttack(), "L1FMNAttack"),
    (L2FMNAttack(), "L2FMNAttack"),
    (LInfFMNAttack(), "LInfFMNAttack")
]

# Apply the Fast Minimum Norm (FMN) attack
epsilons = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1, 2, 5, 7, 9]

for attack, attack_name in attacks:

    if attack_name == "L0FMNAttack":
        raw_adversarial, clipped_adversarial, success = attack(fmodel, images, labels, epsilons=None)
        num_epsilons = 1
    else:
        # Perform the attack with multiple epsilons
        raw_adversarial, clipped_adversarial, success = attack(fmodel, images, labels, epsilons=epsilons)
        num_epsilons = len(epsilons)

    # Analyze the results
    print(f"\nResults for {attack_name}:")
    if attack_name == "L0FMNAttack":
        acc = fb.accuracy(fmodel, clipped_adversarial, labels)
        perturbation_sizes = (clipped_adversarial - images).norms.l0(axis=(1, 2, 3)).numpy()
        print(f"  Attack: Accuracy after attack: {acc * 100:4.1f} %, Perturbation size: {perturbation_sizes}")
    else:
        for eps, advs_, succ_ in zip(epsilons, clipped_adversarial, success):
            acc = fb.accuracy(fmodel, advs_, labels)
            perturbation_sizes = (advs_ - images).norms.l2(axis=(1, 2, 3)).numpy()
            print(
                f"  Epsilon {eps:<6}: Accuracy after attack: {acc * 100:4.1f} %, Success: {succ_}, Perturbation size: {perturbation_sizes}")

    # Number of images in the batch
    num_images = len(images)

    # Set up the figure to visualize the grid
    fig, axes = plt.subplots(num_images, num_epsilons + 1 if attack_name != "L0FMNAttack" else 2, figsize=(15, 15))
    fig.suptitle(f'Adversarial Examples for {attack_name}', fontsize=20)

    # Loop through each sample and each epsilon value
    for i in range(num_images):
        # Plot the original image in the first column
        ax = axes[i, 0]
        original_img = images[i].numpy().squeeze()
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
                    ax.set_title(f'Eps: {epsilons[j]}', fontsize=10)
                ax.axis('off')

    # Remove the space between subplots to make them more compact
    plt.subplots_adjust(wspace=0, hspace=0)

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'resources/adversarial_grid_{attack_name}.png')
    plt.show()

