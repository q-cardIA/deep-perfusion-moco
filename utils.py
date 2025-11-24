import pathlib
import re
from typing import Tuple

import numpy as np
import pydicom
import sklearn.decomposition
from matplotlib import pyplot as plt
from skimage.exposure import equalize_hist
from skimage.transform import resize


def preprocess_image(images, CI, hist_eq):
    """
    Preprocess images by cropping, normalizing and histogram equalizing.

    This is the primary preprocessing function for dl_moco's motion correction
    pipeline. It prepares raw MRI slices for input to the first affine
    registration model by applying spatial cropping, intensity normalization,
    and optional histogram equalization.

    Process:
        1. Crop each 2D frame to specified region of interest
        2. Normalize intensities to [0, 1] range per frame
        3. Optionally apply histogram equalization for contrast enhancement
        4. Resize output to fixed 128×128 dimensions

    Args:
        images (np.ndarray): 3D array of shape (height, width, time_frames)
                           containing raw MRI temporal sequence.
        CI (list or tuple): Crop indices [y_min, y_max, x_min, x_max] defining
                          the spatial region to extract from each frame.
        hist_eq (bool): If True, apply histogram equalization to enhance
                       contrast. dl_moco always uses hist_eq=True.

    Returns:
        np.ndarray: Preprocessed 3D array of shape (time_frames, 128, 128)
                   with normalized and optionally equalized intensities.
                   Note: dimension order is transposed from input.

    Notes:
        - Used by dl_moco as first step before affine registration
        - Output shape is (T, 128, 128) vs input (H, W, T)
        - Normalization is per-frame to handle intensity variations
        - Histogram equalization improves feature detection for registration

    """
    # Make empty numpy array
    normalized_LR = np.empty((images.shape[2], 128, 128))
    for x in range(images.shape[2]):
        image_2d = images[:, :, x]
        image_2d = image_2d[CI[0] : CI[1], CI[2] : CI[3]]
        image_2d = (image_2d - np.min(image_2d)) / (np.max(image_2d) - np.min(image_2d))
        if hist_eq:
            normalized_LR[x, :, :] = equalize_hist(image_2d)
        else:
            normalized_LR[x, :, :] = image_2d
    return normalized_LR


def pca_ref(imgs, nComps):
    """Generate PCA reference features from temporal image series.

    This function performs Principal Component Analysis (PCA) on a temporal
    sequence of 2D images to extract dominant temporal features. Used in
    dl_moco for creating feature representations that capture the main
    patterns of variation across time frames.

    Process:
        1. Mean-center each time frame independently
        2. Reshape images to 2D matrix (pixels × time)
        3. Fit PCA and project to reduced dimensionality
        4. Reconstruct images using only first nComps components
        5. Restore original mean values

    Args:
        imgs (np.ndarray): 3D array of shape (height, width, time_frames)
                          containing temporal sequence of images.
        nComps (int): Number of principal components to retain. Typical values:
                     - 3 components for second affine stage in dl_moco
                     - 2 components for non-rigid stage in dl_moco

    Returns:
        np.ndarray: Reconstructed 3D array with same shape as input (height,
                   width, time_frames), containing PCA-filtered images that
                   preserve only the nComps strongest temporal patterns.

    Notes:
        - Used by dl_moco to generate references for registration models
        - Modifies input array in-place during mean-centering
    """
    T = imgs.shape[2]
    mu = np.zeros(T)
    for t in range(T):
        temp = imgs[:, :, t]
        mu[t] = np.mean(temp)
        imgs[:, :, t] = temp - mu[t]

    imgs1 = imgs.reshape(imgs.shape[0] * imgs.shape[1], imgs.shape[2])

    pca = sklearn.decomposition.PCA()
    pca.fit(imgs1)

    new_imgs = np.dot(pca.transform(imgs1)[:, :nComps], pca.components_[:nComps, :])

    new_imgs = new_imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2])

    for t in range(T):
        new_imgs[:, :, t] = new_imgs[:, :, t] + mu[t]
    return new_imgs


def normalise(the_images):
    """Normalize 3D image series to [0, 1] range per time frame.

    Performs frame-by-frame intensity normalization by scaling each 2D slice
    independently to the [0, 1] range based on its own min/max values.
    Used by dl_moco before the second affine registration stage to ensure
    consistent intensity ranges after the first warping operation.

    Process:
        1. Find min/max intensity for each frame independently
        2. Scale each frame: (image - min) / (max - min)
        3. Preserve temporal dimension ordering

    Args:
        the_images (np.ndarray): 3D array of shape (height, width, time_frames)
                                containing image sequence with arbitrary
                                intensity ranges.

    Returns:
        np.ndarray: Normalized 3D array with same shape, where each frame's
                   intensities are scaled to [0, 1] range independently.

    Notes:
        - Used by dl_moco after first affine warping and before second stage
        - Each time frame normalized independently to handle temporal variations
        - Paired with histogram_equalize in dl_moco pipeline
        - Prevents intensity drift across registration stages
    """
    min_values = np.min(the_images, axis=(0, 1))
    min_image = np.tile(min_values, (the_images.shape[0], the_images.shape[1], 1))
    max_values = np.max(the_images, axis=(0, 1))
    max_image = np.tile(max_values, (the_images.shape[0], the_images.shape[1], 1))
    return (the_images - min_image) / (max_image - min_image)


def histogram_equalize(images):
    """
    Histogram equalization for 3D images.

    Applies adaptive histogram equalization to each 2D frame independently
    to enhance local contrast. Used by dl_moco after normalization in the
    second affine stage to improve feature visibility for registration.

    Process:
        1. Apply scikit-image's equalize_hist to each time frame
        2. Enhances contrast by spreading out intensity histogram
        3. Improves registration accuracy by making features more distinct

    Args:
        images (np.ndarray): 3D array of shape (height, width, time_frames)
                           containing normalized images (typically [0, 1] range).

    Returns:
        np.ndarray: Histogram-equalized 3D array with same shape as input.
                   Each frame has enhanced contrast with improved dynamic range.

    Notes:
        - Used by dl_moco in combination with normalise before second affine stage
        - Also used in preprocess_image for initial preprocessing
        - Enhances edges and features critical for image registration
        - Each frame processed independently to handle temporal intensity changes
        - Uses scikit-image's adaptive histogram equalization (equalize_hist)

    :param images: 3D numpy array of shape (nr_x_coord, nr_y_coord, nr_image_frames)
    :return: 3D numpy array of shape (nr_x_coord, nr_y_coord, nr_image_frames)
    """
    hist_equalized_images = np.empty(images.shape)
    for x in range(images.shape[2]):
        image_2d = images[:, :, x]
        hist_equalized_images[:, :, x] = equalize_hist(image_2d)
    return hist_equalized_images
