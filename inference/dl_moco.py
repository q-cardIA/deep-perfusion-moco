"""
Deep Learning Motion Correction (MOCO) for Stress Perfusion Cardiac MRI.

This module implements a multi-stage motion correction pipeline using three
sequential deep learning models to register cardiac MRI slices. The pipeline
consists of two affine registration stages followed by non-rigid deformation
for fine alignment.

Dependencies:
    - numpy: Array operations
    - onnxruntime: ONNX model inference
    - torch: PyTorch tensor operations
    - monai: Medical image processing (Warp layer)
    - utils: Custom utilities for preprocessing and PCA

Model Pipeline:
    1. First affine model: Initial coarse alignment
    2. Second affine model: Refined affine registration with PCA reference
    3. Non-rigid model: Fine non-rigid deformation field estimation
"""

from pathlib import Path

import numpy as np
import onnxruntime as rt
import torch
from monai.networks.blocks import Warp

import utils

# Directory containing trained ONNX models
MODEL_DIR = Path("models")

# Model file paths
MODEL1 = "first_MRI-affine-295.onnx"  # First stage affine registration
MODEL2 = "second_affine.onnx"  # Second stage affine registration
MODEL3 = "non-rigid_config44-mri_with_noise.onnx"  # Non-rigid deformation


def do_moco(the_slices, the_crop_inds):
    """
    Perform motion correction on cardiac MRI slices using deep learning.

    This function applies a three-stage registration pipeline to correct motion
    artifacts in cardiac MRI time series data. Each slice is independently
    registered to a reference frame (10th frame from the end).

    Args:
        the_slices (dict): Dictionary mapping slice identifiers to 3D numpy arrays
                          with shape (height, width, time_frames). Each array
                          represents a temporal sequence of 2D MRI slices.
        the_crop_inds (tuple): Tuple of 4 integers (y_start, y_end, x_start, x_end)
                              defining the spatial crop region to extract from
                              original images before registration.

    Returns:
        dict: Dictionary with same keys as the_slices, containing motion-corrected
              3D arrays with shape (height, width, time_frames). The spatial
              dimensions are reduced by 32 pixels (16 on each side) due to
              cropping in the non-rigid stage.

    Pipeline stages:
        1. Preprocessing: Crop, normalize, and histogram equalize input images
        2. First affine: Register to reference frame (frame -10)
        3. Apply first deformation to original images
        4. Second affine: Refine registration using PCA features (3 components)
        5. Crop to 96x96 for non-rigid model
        6. Non-rigid: Final deformation field estimation with PCA (2 components)
        7. Apply cumulative transformations to original images

    Notes:
        - Uses bilinear interpolation with border padding for warping
        - Reference frame (for first registration) is the 10th frame from the end (-10 index)
        - Models are loaded with CPU execution provider
        - Final output is resized to 96x96 (from original 128x128 after crop)
    """

    # Load ONNX models for each registration stage
    ort_session1 = rt.InferenceSession(
        str(MODEL_DIR / MODEL1), providers=["CPUExecutionProvider"]
    )
    ort_session2 = rt.InferenceSession(
        str(MODEL_DIR / MODEL2), providers=["CPUExecutionProvider"]
    )
    ort_session3 = rt.InferenceSession(
        str(MODEL_DIR / MODEL3), providers=["CPUExecutionProvider"]
    )

    # Initialize warp layer for applying deformation fields
    warp_layer = Warp("bilinear", "border")

    # Dictionary to store motion-corrected slices
    moco_slices = {}

    # Process each slice independently
    for slice_key in the_slices:
        # Preprocess: crop, normalize, and histogram equalize
        im_reg = utils.preprocess_image(
            the_slices[slice_key], the_crop_inds, hist_eq=True
        )

        # Create reference frame array (using 10th frame from end)
        reference_array = np.repeat(
            im_reg[-10, :, :][np.newaxis, ...], im_reg.shape[0], axis=0
        )

        # Extract original cropped image for warping
        im_orig = the_slices[slice_key][
            the_crop_inds[0] : the_crop_inds[1], the_crop_inds[2] : the_crop_inds[3]
        ]

        # === STAGE 1: First affine registration ===
        # Prepare input: concatenate moving and reference images
        ort_inputs = {
            ort_session1.get_inputs()[0]
            .name: np.concatenate(
                (im_reg[:, np.newaxis, ...], reference_array[:, np.newaxis, ...]),
                axis=1,
            )
            .astype(np.float32)
        }
        ort_outs = ort_session1.run(None, ort_inputs)
        ddf_ = ort_outs[0]  # Dense deformation field from first model

        # Apply first affine transformation to original images
        im_orig_transposed = np.transpose(im_orig, (2, 0, 1))
        pred_image_orig = warp_layer(
            torch.from_numpy(im_orig_transposed[:, np.newaxis, ...].astype(np.float32)),
            torch.from_numpy(ddf_.astype(np.float32)),
        ).numpy()[:, 0, :, :]

        im_orig = warp_layer(
            torch.from_numpy(im_orig_transposed[:, np.newaxis, ...].astype(np.float32)),
            torch.from_numpy(ddf_.astype(np.float32)),
        ).numpy()[:, 0, :, :]

        # === STAGE 2: Second affine registration with PCA features ===
        # Generate PCA features (3 components) from warped images
        pca_data = utils.pca_ref(np.transpose(pred_image_orig, (1, 2, 0)), 3)
        pca_data = utils.histogram_equalize(utils.normalise(pca_data))

        # Normalize and equalize warped images for second stage
        pred_image_orig = np.transpose(
            utils.histogram_equalize(
                utils.normalise(np.transpose(pred_image_orig, (1, 2, 0)))
            ),
            (2, 0, 1),
        )

        # Prepare input: concatenate warped images with PCA features
        ort_inputs2 = {
            ort_session2.get_inputs()[0]
            .name: np.concatenate(
                (
                    pred_image_orig[:, np.newaxis, ...],
                    np.transpose(pca_data, (2, 0, 1))[:, np.newaxis, ...],
                ),
                axis=1,
            )
            .astype(np.float32)
        }
        ort_outs2 = ort_session2.run(None, ort_inputs2)
        ddf_ = ort_outs2[0]  # Second deformation field

        # Apply second affine transformation
        pred_image_orig = warp_layer(
            torch.from_numpy(pred_image_orig[:, np.newaxis, ...].astype(np.float32)),
            torch.from_numpy(ddf_.astype(np.float32)),
        ).numpy()[:, 0, :, :]

        im_orig = warp_layer(
            torch.from_numpy(im_orig[:, np.newaxis, ...].astype(np.float32)),
            torch.from_numpy(ddf_.astype(np.float32)),
        ).numpy()[:, 0, :, :]

        # === STAGE 3: Non-rigid registration ===
        # Crop to 96x96 (remove 16 pixels from each side)
        pred_image_orig = pred_image_orig[:, 16:-16, 16:-16]
        im_orig = im_orig[:, 16:-16, 16:-16]

        # Generate PCA features (2 components) for non-rigid stage
        pca_data = utils.pca_ref(np.transpose(pred_image_orig, (1, 2, 0)), 2)

        # Prepare input for non-rigid model
        ort_inputs3 = {
            ort_session3.get_inputs()[0]
            .name: np.concatenate(
                (
                    pred_image_orig[:, np.newaxis, ...],
                    np.transpose(pca_data, (2, 0, 1))[:, np.newaxis, ...],
                ),
                axis=1,
            )
            .astype(np.float32)
        }
        ort_outs3 = ort_session3.run(None, ort_inputs3)
        ddf_ = ort_outs3[0]  # Non-rigid deformation field

        # Apply final non-rigid transformation
        im_orig = warp_layer(
            torch.from_numpy(im_orig[:, np.newaxis, ...].astype(np.float32)),
            torch.from_numpy(ddf_.astype(np.float32)),
        ).numpy()[:, 0, :, :]

        # Store motion-corrected slice (transpose back to H x W x T)
        moco_slices[slice_key] = np.transpose(im_orig, (1, 2, 0))

    return moco_slices
