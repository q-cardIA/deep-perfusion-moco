from pathlib import Path

import numpy as np
import onnxruntime as rt
import torch
import utils
from monai.networks.blocks import Warp

MODEL_DIR = Path("models")
# Specify model locations
MODEL1 = "first_MRI-affine-295.onnx"
MODEL2 = "second_affine.onnx"
MODEL3 = "non-rigid_config44-mri_with_noise.onnx"


def do_moco(the_slices, the_crop_inds):

    ort_session1 = rt.InferenceSession(
        str(MODEL_DIR / MODEL1), providers=["CPUExecutionProvider"]
    )
    ort_session2 = rt.InferenceSession(
        str(MODEL_DIR / MODEL2), providers=["CPUExecutionProvider"]
    )
    ort_session3 = rt.InferenceSession(
        str(MODEL_DIR / MODEL3), providers=["CPUExecutionProvider"]
    )

    # Define warp layer and metrics
    warp_layer = Warp("bilinear", "border")
    moco_slices = {}
    for slice in the_slices:
        im_reg = utils.preprocess_image(the_slices[slice], the_crop_inds, hist_eq=True)
        reference_array = np.repeat(
            im_reg[-10, :, :][np.newaxis, ...], im_reg.shape[0], axis=0
        )
        im_orig = the_slices[slice][
            the_crop_inds[0] : the_crop_inds[1], the_crop_inds[2] : the_crop_inds[3]
        ]
        ort_inputs = {
            ort_session1.get_inputs()[0]
            .name: np.concatenate(
                (im_reg[:, np.newaxis, ...], reference_array[:, np.newaxis, ...]),
                axis=1,
            )
            .astype(np.float32)
        }
        ort_outs = ort_session1.run(None, ort_inputs)
        ddf_ = ort_outs[0]

        pred_image_orig = warp_layer(
            torch.from_numpy(
                np.transpose(im_orig, (2, 0, 1))[:, np.newaxis, ...].astype(np.float32)
            ),
            torch.from_numpy(ddf_.astype(np.float32)),
        ).numpy()[:, 0, :, :]

        im_orig = warp_layer(
            torch.from_numpy(
                np.transpose(im_orig, (2, 0, 1))[:, np.newaxis, ...].astype(np.float32)
            ),
            torch.from_numpy(ddf_.astype(np.float32)),
        ).numpy()[:, 0, :, :]

        pca_data = utils.pca_ref(np.transpose(pred_image_orig, (1, 2, 0)), 3)
        pca_data = utils.histogram_equalize(utils.normalise(pca_data))

        pred_image_orig = np.transpose(
            utils.histogram_equalize(
                utils.normalise(np.transpose(pred_image_orig, (1, 2, 0)))
            ),
            (2, 0, 1),
        )

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
        ddf_ = ort_outs2[0]

        pred_image_orig = warp_layer(
            torch.from_numpy(pred_image_orig[:, np.newaxis, ...].astype(np.float32)),
            torch.from_numpy(ddf_.astype(np.float32)),
        ).numpy()[:, 0, :, :]

        im_orig = warp_layer(
            torch.from_numpy(im_orig[:, np.newaxis, ...].astype(np.float32)),
            torch.from_numpy(ddf_.astype(np.float32)),
        ).numpy()[:, 0, :, :]

        # Resize to 96x96 for non-rigid model
        pred_image_orig = pred_image_orig[:, 16:-16, 16:-16]
        im_orig = im_orig[:, 16:-16, 16:-16]

        pca_data = utils.pca_ref(np.transpose(pred_image_orig, (1, 2, 0)), 2)
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
        ddf_ = ort_outs3[0]

        im_orig = warp_layer(
            torch.from_numpy(im_orig[:, np.newaxis, ...].astype(np.float32)),
            torch.from_numpy(ddf_.astype(np.float32)),
        ).numpy()[:, 0, :, :]

        moco_slices[slice] = np.transpose(im_orig, (1, 2, 0))

    return moco_slices
