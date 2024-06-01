# %%
from skimage import transform
from skimage.filters import gaussian
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def build_laplacian_pyramid(image, max_levels=5):
    pyramid = []
    current_layer = image
    for _ in range(max_levels):
        # Apply Gaussian filter and downsample
        gaussian_layer = gaussian(current_layer, sigma=2, mode='reflect')
        downsampled = transform.resize(gaussian_layer, (current_layer.shape[0] // 2, current_layer.shape[1] // 2))

        # Upsample and subtract to get the Laplacian
        upsampled = transform.resize(downsampled, current_layer.shape)
        laplacian = current_layer - upsampled
        pyramid.append(laplacian)

        # Update the current layer
        current_layer = downsampled

    pyramid.append(current_layer)
    return pyramid

def reconstruct_from_laplacian_pyramid(pyramid):
    reconstructed_image = pyramid[-1]
    for laplacian in reversed(pyramid[:-1]):
        # Upsample the current image
        upsampled = transform.resize(reconstructed_image, laplacian.shape)
        # Add the Laplacian layer
        reconstructed_image = upsampled + laplacian
    return reconstructed_image


# %%
original_image = fits.open('/Users/danny/Desktop/cos0_Set1_rotate1_area1_37_kappa.fits')[0].data
original_image = original_image[0:512, 0:512]

# Generate the Laplacian Pyramid
laplacian_pyramid = build_laplacian_pyramid(original_image, max_levels=5)

# Reconstruct the image from the pyramid
reconstructed_image = reconstruct_from_laplacian_pyramid(laplacian_pyramid)

# Display the original and reconstructed images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image)
plt.title('Reconstructed Image')
plt.show()

# Check if the images are exactly the same
is_identical = np.allclose(original_image, reconstructed_image)
is_identical

# %%
def visualize_laplacian_pyramid(pyramid):
    # Plot for the Laplacian Pyramid
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, layer in enumerate(pyramid):
        row, col = divmod(i, 3)
        im = axes[row, col].imshow(layer, cmap='viridis')
        axes[row, col].set_title(f'Layer {i}')
        fig.colorbar(im, ax=axes[row, col], orientation='horizontal', pad=0.1)

    # Fill any empty subplots
    for j in range(i+1, 6):
        row, col = divmod(j, 3)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_reconstruction_process(pyramid, reconstructed_image):
    num_layers = len(pyramid)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    temp_reconstructed = pyramid[-1]
    for i, layer in enumerate(reversed(pyramid[:-1])):
        row, col = divmod(i, 3)
        temp_reconstructed = transform.resize(temp_reconstructed, layer.shape) + layer
        im = axes[row, col].imshow(temp_reconstructed, cmap='viridis')
        axes[row, col].set_title(f'Reconstruction {i}')
        fig.colorbar(im, ax=axes[row, col], orientation='horizontal', pad=0.1)

    # Final reconstructed image
    row, col = divmod(num_layers - 1, 3)
    im = axes[row, col].imshow(reconstructed_image, cmap='viridis')
    axes[row, col].set_title('Final Reconstruction')
    fig.colorbar(im, ax=axes[row, col], orientation='horizontal', pad=0.1)

    # Fill any empty subplots
    for j in range(num_layers, 6):
        row, col = divmod(j, 3)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize the Laplacian pyramid and the reconstruction process separately
visualize_laplacian_pyramid(laplacian_pyramid)
visualize_reconstruction_process(laplacian_pyramid, reconstructed_image)

# %%
# kornia version for 4D tensor operations

import torch
from kornia.geometry.transform import resize
from kornia.filters.gaussian import gaussian_blur2d

import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt

shear = fits.open('/Users/danny/Desktop/cos0_Set1_rotate1_area1_37_gamma1.fits')[0].data[0:512, 0:512]
kappa = fits.open('/Users/danny/Desktop/cos0_Set1_rotate1_area1_37_kappa.fits')[0].data[0:512, 0:512]
t1 = torch.tensor(np.float32(shear)).view(1, 1, 512, 512)
t2 = torch.tensor(np.float32(kappa)).view(1, 1, 512, 512)
t = torch.concat([t1, t2])

def build_laplacian_pyramid(tensor, max_levels=5):
    pyramid = []
    current_layer = tensor
    for _ in range(max_levels):
        # Apply Gaussian filter and downsample
        gaussian_layer = gaussian_blur2d(current_layer, kernel_size=(5, 5), sigma=(2., 2.), border_type='reflect')
        downsampled = resize(gaussian_layer, size=(current_layer.shape[-2]//2, current_layer.shape[-1]//2))
        # Upsample and subtract to get the Laplacian
        upsampled = resize(downsampled, size=current_layer.shape[-2:])
        laplacian = current_layer - upsampled
        pyramid.append(laplacian)
        # Update the current layer
        current_layer = downsampled
    pyramid.append(current_layer)
    return pyramid


lap_pyramid = build_laplacian_pyramid(t, max_levels=5)

x = lap_pyramid[-1]
for laplacian in reversed(lap_pyramid[:-1]):
    # Upsample the current image
    upsampled = resize(x, size=laplacian.shape[-2:])
    # Add the Laplacian layer
    x = upsampled + laplacian


# %%

def visualize_laplacian_pyramid(pyramid):
    # Plot for the Laplacian Pyramid
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, layer in enumerate(pyramid):
        row, col = divmod(i, 3)
        im = axes[row, col].imshow(layer[0][0], cmap='viridis')
        axes[row, col].set_title(f'Layer {i}')
        fig.colorbar(im, ax=axes[row, col], orientation='horizontal', pad=0.1)

    # Fill any empty subplots
    for j in range(i+1, 6):
        row, col = divmod(j, 3)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_reconstruction_process(pyramid, reconstructed_image):
    num_layers = len(pyramid)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    temp_reconstructed = pyramid[-1]
    for i, layer in enumerate(reversed(pyramid[:-1])):
        row, col = divmod(i, 3)
        temp_reconstructed = resize(temp_reconstructed, layer.shape[-2:]) + layer
        im = axes[row, col].imshow(temp_reconstructed[0][0], cmap='viridis')
        axes[row, col].set_title(f'Reconstruction {i}')
        fig.colorbar(im, ax=axes[row, col], orientation='horizontal', pad=0.1)

    # Final reconstructed image
    row, col = divmod(num_layers - 1, 3)
    im = axes[row, col].imshow(reconstructed_image[0][0], cmap='viridis')
    axes[row, col].set_title('Final Reconstruction')
    fig.colorbar(im, ax=axes[row, col], orientation='horizontal', pad=0.1)

    # Fill any empty subplots
    for j in range(num_layers, 6):
        row, col = divmod(j, 3)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


visualize_laplacian_pyramid(lap_pyramid)
visualize_reconstruction_process(lap_pyramid, x)

# %%
