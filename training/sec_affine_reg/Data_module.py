# Import modules
from Dataset import dataset_reference_affine

# Import libraries
import os
import monai
import torch
import random
import numpy as np
from monai import transforms
from monai.data import DataLoader
from monai.transforms import Compose

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

# Setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class DataModule: 
    def __init__(self, config):
        self.config = config
        self.datasets = {"train": None, "valid": None}
        self.csv_loc = config["dataloader"]["csv_loc"] 
        self._build_datasets()

    def train_dataloader(self):
        return self._dataloader("train")
    
    def valid_dataloader(self):
        return self._dataloader("valid")

    def test_dataloader(self):
        return self._dataloader("test")
    
    def _dataloader(self, dataloader_type):
        """Returns the dataloader based on the dataloader_type '(train, valid, test)')"""
        random_indices = random.sample(range(len(self.datasets[dataloader_type])), len(self.datasets[dataloader_type]) // 2)
        dataset = torch.utils.data.Subset(self.datasets[dataloader_type], random_indices)

        return DataLoader(
            dataset=dataset, 
            batch_size = self.config["dataloader"][dataloader_type]["batch_size"],
            drop_last = self.config["dataset"][dataloader_type]["drop_last"],
            shuffle = self.config["dataset"][dataloader_type]["shuffle"],
            num_workers= self.config["general"]["num_workers"],
            pin_memory=True,
        )

    def _dataset(self, csv_loc, key):
        """Returns the dataset based on the csv_loc. 'ref_shift' indicates the reference frame (LV_peak + ref_shift)
        Dataset returns frame i and reference frame (fixed per patient).
        """
        # If we apply augmentations, this value is true
        if self.config["augmentation"]["apply"]:
            return dataset_reference_affine(csv_loc, self.config["training"]["hist_eq"], self.config["training"]["include_lowres"], key=key, config = self.config, transform=self.compose_transformation())
        else:
            return dataset_reference_affine(csv_loc, self.config["training"]["hist_eq"], self.config["training"]["include_lowres"], key=key, config = self.config, transform=None)
    
    def _build_datasets(self):
        """Builds the datasets for train, valid and test. Saved in self.datasets"""
        for key in self.datasets:
            csv_loc = self.csv_loc[key]
            dataset = self._dataset(csv_loc, key)
            self.datasets[key] = dataset

    def compose_transformation(self):
        """ Composes augmentations to apply to the image, based on information in config"""
        compose_config = self.config["augmentation"]
        trans = Compose([
            transforms.RandAffine(prob=compose_config["affine"]["prob"], rotate_range=[compose_config["affine"]["rotation"]["range"]], shear_range=0, translate_range=[compose_config["affine"]["translation"]["range"]], scale_range=0, spatial_size=0),
            transforms.RandScaleIntensity(factors=compose_config["scale_intensity"]["factors"], prob=compose_config["scale_intensity"]["prob"]),
            transforms.RandShiftIntensity(offsets=compose_config["shift_intensity"]["offsets"], prob=compose_config["shift_intensity"]["prob"]),
            transforms.RandGaussianNoise(prob=compose_config["gaussian_noise"]["prob"], mean=compose_config["gaussian_noise"]["mean"], std=compose_config["gaussian_noise"]["std"], dtype=np.float32)
        ])

        return trans