from monai.data import Dataset
from monai import transforms
import torch
import random
import os
import re
import numpy as np
import pandas as pd
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
from skimage.exposure import equalize_adapthist, equalize_hist

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


class dataset_reference_affine(Dataset): 
    """Our dataset. This class inherits Dataset and overrides the following methods: 
    1. '__len__' = size of dataset
    2. '__getitem__' = dataset[i] returns ith sample (reads images)

    Based on: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, csv_file, apply_hist_eq, include_lowres, key, config, transform=None):
        """
        Arguments:
            csv_file (string): path to csv file 
            root_dir (string): directory with all images
            ref_shift (int): value to indicate which frame is reference (LV_peak + ref_shift)
        """
        self.csv_file = csv_file
        self.key = key # train, valid or test
        self.config = config

        # List with unique values=range(self.data.keys()). Used to find corresponding image series and reference image
        self.image_per_key = [] 

        # Not used yet
        self.lowres_data = {}
        self.include_lowres = include_lowres

        # Create dictionary with key=patient_nr, value=[list_image_locations],reference_index
        self.data = self.create_image_list()
        
        # Transformations / augmentations
        self.transform = transform
        # self._seed = 0 # seed to make sure same transformation is applied to image and reference, will be randomized for every __getitem__
        
        # Histogram equalization
        self.apply_hist_eq = apply_hist_eq

    def __len__(self):      
        """ Return number of images present in this data set""" 
        return len(self.image_per_key)

    def __getitem__(self, idx):
        """
        Function returns the sample of index idx. Sample = {'image':image_array, 'reference':reference_array}
        """
        # make sure idx is a list of indics
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Access correct key, based on this patient_ind
        patient_key = self.image_per_key[idx]
        lst_path = self.data[patient_key]

        # Get index of idx with in patient data
        first_patient_ind = self.image_per_key.index(patient_key) # find first occurence of patient_id (0 t/m max nr image series)
        patient_specific_ind = idx - first_patient_ind  # index of image within specific patient data (so within dictionary list)
        image_path = lst_path[patient_specific_ind]     
        
        # Get reference image path
        reference_path = image_path.replace("img", "pca")
        # Load moving and fixed image
        image_arr = np.load(image_path)                 
        reference_arr = np.load(reference_path)

        # Save original images
        image_arr_org = image_arr.copy()
        reference_arr_org = reference_arr.copy()

        # Normalize images (range of 0-1)
        image_arr = (image_arr - np.min(image_arr)) / (np.max(image_arr) - np.min(image_arr))
        reference_arr = (reference_arr - np.min(reference_arr)) / (np.max(reference_arr) - np.min(reference_arr))

        if self.apply_hist_eq:
            # Histogram equalization
            image_arr = equalize_hist(image_arr)
            reference_arr = equalize_hist(reference_arr)

        # Apply augmentations
        if self.transform != None and self.key != "valid":     
            # Convert shape of image to (bs, nr_channel, x, y)
            img_new = np.empty((1,128,128))
            img_new[0,:,:] = image_arr
            
            image_arr = self.transform(img_new)[0]

        
        sample = {'image':image_arr, 'reference':reference_arr, "image_original":image_arr_org, "reference_original":reference_arr_org}

        return sample

    def create_image_list(self):
        """
        Function creates dictionary of image locations like: 
            {patient 1_1: ([list of image locations], index_LV_peak), patient_1_2: ([list of image locations], index_LV_peak), etc} 
        It excludes the low-resolution images based on the low_res value

        Arguments
            ref_shift (int): indicating the reference frame compared to the LV_peak index
        
        """
        info_dict = {}
        
        patient_info = pd.read_csv(self.csv_file, header=None)

        # Determine how many images are included for this patient
        for index, row in patient_info.iterrows():
            # Get info from csv
            dir_path, LV_peak, crop_inds, low_res, low_res_recognition = row.tolist()[0], row.tolist()[1], row.tolist()[2:6], row.tolist()[6], row.tolist()[7]

            # Get the unique image series per patient (4 image series)
            unique_image_series = list({filename[-11:] for filename in os.listdir(dir_path)})

            for iss, image_series in enumerate(unique_image_series):
                image_series_images = [] # first outside this
                # Define unique key per image series
                key = "patient_{}_{}".format(index, iss+1)

                # If we include all images: 
                if self.include_lowres:
                    for f, filename in enumerate(os.listdir(dir_path)): # include all images of the current image series
                        if image_series in filename and "result" in filename:
                            image_series_images.append(os.path.join(dir_path, filename))
                            self.image_per_key.append(key)
                    # When we have all image_paths, add it to dictionary
                    info_dict[key] = image_series_images

                # If we exclude low-resolution images:
                elif str(low_res_recognition) not in image_series:
                    for f, filename in enumerate(os.listdir(dir_path)): 
                        if image_series in filename:
                            image_series_images.append(os.path.join(dir_path, filename))
                            self.image_per_key.append(key)
                    # When we have all image_paths, add it to dictionary
                    info_dict[key] = image_series_images

        return info_dict

    def create_transform(self):
        """
        Function randomly creates an affine transformation
        Depends on:
         - self.transform: whether transformation should be applied or not
         - self.key: whether we are in train, valid or test mode. Only in training mode, we return a transformation
        """
        if self.transform != None and self.key != "valid":    
            random_rotate_param = np.random.uniform(self.config["augmentation"]["affine"]["rotation"]["range"][0], self.config["augmentation"]["affine"]["rotation"]["range"][1])
            random_translate_param = np.random.randint(self.config["augmentation"]["affine"]["translation"]["range"][0], self.config["augmentation"]["affine"]["translation"]["range"][1])
            return transforms.Affine(rotate_params=random_rotate_param, shear_params=0, translate_params=random_translate_param, scale_params=0)
    
        else:
            return 0
        
