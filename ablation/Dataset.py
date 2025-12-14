from monai.data import Dataset
import torch
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
    def __init__(self, csv_file, apply_hist_eq, include_lowres, key, cut_edge=False, transform=None):
        """
        Arguments:
            csv_file (string): path to csv file 
            root_dir (string): directory with all images
            ref_shift (int): value to indicate which frame is reference (LV_peak + ref_shift)
        """
        self.csv_file = csv_file
        self.key = key # train, valid or test

        # List with unique values=range(self.data.keys()). Used to find corresponding image series and reference image
        self.image_per_key = [] 

        # Not used yet
        self.lowres_data = {}
        self.include_lowres = include_lowres

        # Create dictionary with key=patient_nr, value=[list_image_locations],reference_index
        self.data = self.create_image_list()
        
        # Transformations / augmentations
        self.transform = transform
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
        lst_path, reference_ind = self.data[patient_key]      

        # Get index of idx with in patient data
        first_patient_ind = self.image_per_key.index(patient_key) # find first occurence of patient_id (0 t/m max nr image series)
        patient_specific_ind = idx - first_patient_ind  # index of image within specific patient data (so within dictionary list)
        image_path = lst_path[patient_specific_ind]     

        # Search for 'imgL{4digit_nr} and replace it with reference_index
        match = re.search(r'imgL(\d{4})', image_path)
        if match:
            original_number = match.group(1)
            ref_number = "0" * (4 - len(str(reference_ind))) + str(reference_ind)
            reference_path = image_path.replace(original_number, ref_number)

        # Load moving and fixed image
        image_arr = np.load(image_path)                 
        reference_arr = np.load(reference_path)
        
        # Normalize images (range of 0-1)
        image_arr = (image_arr - np.min(image_arr)) / (np.max(image_arr) - np.min(image_arr))
        reference_arr = (reference_arr - np.min(reference_arr)) / (np.max(reference_arr) - np.min(reference_arr))
        image_arr_org, reference_arr_org = image_arr, reference_arr

        if self.apply_hist_eq:
            # Histogram equalization
            image_arr = equalize_hist(image_arr)
            reference_arr = equalize_hist(reference_arr)

        # Apply augmentations
        if self.transform != None and self.key != "valid": 
            # Convert shape of image to (bs, nr_channel, x, y)
            img_new = np.empty((1,128,128))
            img_new[0,:,:] = image_arr

            ref_new = np.empty((1,128,128))
            ref_new[0,:,:] = reference_arr
            
            image_arr = self.transform(img_new)[0]
            reference_arr = self.transform(ref_new)[0]
        
        sample = {'image':image_arr, 'reference':reference_arr, 'image_org':image_arr_org, 'reference_org':reference_arr_org}

        return sample

    def create_image_list(self):
        """
        Function creates dictionary of image locations like: 
            {patient 1_1: ([list of image locations], index_LV_peak), patient_1_2: ([list of image locations], index_LV_peak), etc} 
        It excludes the low-resolution images based on the low_res value
        
        """
        info_dict = {}
        
        patient_info = pd.read_csv(self.csv_file, header=None)

        # Determine how many images are included for this patient
        for index, row in patient_info.iterrows():
            # Get info from csv
            dir_path, LV_peak, crop_inds, low_res, low_res_recognition = row.tolist()[0], row.tolist()[1], row.tolist()[2:6], row.tolist()[6], row.tolist()[7]

            # Get the unique image series per patient (4 image series) --> depends on how the data is saved
            unique_image_series = list({filename[-11:] for filename in os.listdir(dir_path)})

            for ims, image_series in enumerate(unique_image_series):
                image_series_images = []
                # Define unique key per image series
                key = "patient_{}_{}".format(index, ims+1)

                # If we include all images:
                if self.include_lowres:
                    for f, filename in enumerate(os.listdir(dir_path)): # include all images of the current image series
                        if image_series in filename:
                            image_series_images.append(os.path.join(dir_path, filename))
                            self.image_per_key.append(key)
                    # When we have all image_paths, add it to dictionary, together with reference frame index (which is end-10)
                    info_dict[key] = image_series_images, len(image_series_images) - 10

                # If we exclude low-resolution images:
                elif str(low_res_recognition) not in image_series:
                    for f, filename in enumerate(os.listdir(dir_path)): 
                        if image_series in filename:
                            image_series_images.append(os.path.join(dir_path, filename))
                            self.image_per_key.append(key)
                    # When we have all image_paths, add it to dictionary, together with reference frame index (which is end-10)
                    info_dict[key] = image_series_images, len(image_series_images) - 10

        return info_dict
