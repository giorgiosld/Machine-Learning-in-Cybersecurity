import tensorflow as tf
import foolbox as fb
import eagerpy as ep
import numpy as np
import matplotlib.pyplot as plt
from foolbox.attacks import L2FMNAttack

# Load your MNIST model
model_path = "mnist_cnn_model.h5"
model = tf.keras.models.load_model(model_path)

# Define preprocessing used by Foolbox
preprocessing = dict(mean=0.0, std=1.0)  # Assuming the input is normalized in range [0, 1]

# Wrap the model in a Foolbox model
fmodel = fb.TensorFlowModel(model, bounds=(0, 1), preprocessing=preprocessing)

# Load MNIST test data
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0  # Normalize to [0, 1]

# Convert labels to int32 (important for compatibility with Foolbox)
y_test = y_test.astype(np.int32)

# Convert data to EagerPy tensors backed by TensorFlow
images = ep.astensor(tf.convert_to_tensor(x_test[:10]))  # Use a batch of 10 for example
labels = ep.astensor(tf.convert_to_tensor(y_test[:10]))

# Clean accuracy
clean_acc = fb.accuracy(fmodel, images, labels)
print(f"Clean accuracy: {clean_acc * 100:.1f} %")

# Apply the Fast Minimum Norm (FMN) attack
attack = L2FMNAttack()  # You can also experiment with other norms (L0, L1, L2, Linf)
epsilons = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1, 2, 5, 7, 9]

# Perform the attack
raw_adversarial, clipped_adversarial, success = attack(fmodel, images, labels, epsilons=epsilons)

# Analyze the results
print("\nAdversarial sample generation results:")
for eps, advs_, succ_ in zip(epsilons, clipped_adversarial, success):
    acc = fb.accuracy(fmodel, advs_, labels)
    perturbation_sizes = (advs_ - images).norms.l2(axis=(1, 2, 3)).numpy()
    print(f"  Epsilon {eps:<6}: Accuracy after attack: {acc * 100:4.1f} %, Success: {succ_}, Perturbation size: {perturbation_sizes}")

# Number of images in the batch
num_images = len(images)
num_epsilons = len(epsilons)

# Sort the images and labels based on the labels (ascending order from 0 to 9)
sorted_indices = np.argsort(y_test[:num_images])
sorted_images = images[sorted_indices]
sorted_labels = y_test[sorted_indices]

# Set up the figure to visualize the grid
fig, axes = plt.subplots(num_images, num_epsilons + 1, figsize=(12, 12))
fig.suptitle('Adversarial Examples for Different Epsilon Values', fontsize=20)

# Loop through each sample and each epsilon value
for i in range(num_images):
    # Plot the original image in the first column
    ax = axes[i, 0]
    original_img = sorted_images[i].numpy().squeeze()
    ax.imshow(original_img, cmap='gray')
    if i == 0:  # Label the epsilon value on the top row
        ax.set_title(f'Original', fontsize=10)
    ax.axis('off')

    # Plot the adversarial images for each epsilon value in subsequent columns
    for j in range(num_epsilons):
        ax = axes[i, j + 1]
        adversarial_img = clipped_adversarial[j][sorted_indices[i]].numpy().squeeze()
        ax.imshow(adversarial_img, cmap='gray')
        if i == 0:  # Label the epsilon value on the top row
            ax.set_title(f'Eps: {epsilons[j]}', fontsize=10)
        ax.axis('off')

# Remove the space between subplots to make them more compact
plt.subplots_adjust(wspace=0, hspace=0)

# Adjust layout and show plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('resources/adversarial_grid.png')
plt.show()
