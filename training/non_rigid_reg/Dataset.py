import os
import torch
import numpy as np
import pandas as pd
from monai import transforms
from monai.data import Dataset
from skimage.exposure import equalize_hist

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


class dataset_reference_deformable(Dataset): 
    def __init__(self, csv_file, apply_hist_eq, include_lowres, key, affine_trans=0, smooth = 0, transform=None):
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

        # Transformations / augmentations
        self.transform = transform
        self.affine_trans = affine_trans
        self.smooth = smooth

        # Create dictionary with key=patient_nr, value=[list_image_locations],reference_index
        self.data = self.create_image_list()
        
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
        
        # Normalize images
        image_arr = (image_arr - np.min(image_arr)) / (np.max(image_arr) - np.min(image_arr))
        reference_arr = (reference_arr - np.min(reference_arr)) / (np.max(reference_arr) - np.min(reference_arr))

        image_arr_training = image_arr
        reference_arr_training = reference_arr

        if self.apply_hist_eq:
            # Histogram equalization
            image_arr_training = equalize_hist(image_arr_training)
            reference_arr_training = equalize_hist(reference_arr_training)
            
        if self.transform != None and self.key != "valid":
            image_arr_training = self.transform(image_arr_training)
            reference_arr_training = self.transform(reference_arr_training)
        else:
            image_arr_training = image_arr_training
            reference_arr_training = reference_arr_training
        
        sample = {'image_model':image_arr_training, 'reference_model':reference_arr_training, 'image':image_arr, 'reference':reference_arr}

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
                        if image_series in filename and "img" in filename: # and self.does_not_contain_substring(dir_path, dont_include)
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

    def apply_affine_transform(self, image_arr_training, reference_arr_training, image_arr, reference_arr):
        """
        Function applies affine transformation to image and reference image
        Output: 
         - shape model arrays = (1,96,96)
         - shape original arrays = (96,96) (e.g. no histogram equalization, smooth, etc.)
        """
        # Create affine transformation object
        random_rotate_param = np.random.uniform(-0.8, 0.8)
        random_translate_param = np.random.randint(-20, 20)
        transform_affine = transforms.Affine(rotate_params=random_rotate_param, shear_params=0, translate_params=random_translate_param, scale_params=0) 

        # Convert shape of image to (bs, nr_channel, x, y)
        img_new_training = np.empty((1,96,96))
        img_new_training[0,:,:] = image_arr_training

        ref_new_training = np.empty((1,96,96))
        ref_new_training[0,:,:] = reference_arr_training

        img_new = np.empty((1,96,96))
        img_new[0,:,:] = image_arr

        ref_new = np.empty((1,96,96))
        ref_new[0,:,:] = reference_arr

        # Apply affine transformation
        img_new_training = transform_affine(img_new_training)
        ref_new_training = transform_affine(ref_new_training)
        img_new = transform_affine(img_new)
        ref_new = transform_affine(ref_new)

        # Prepare image for next augmentation
        img_new_training = np.asarray(img_new_training[0])
        ref_new_training = np.asarray(ref_new_training[0])

        # Make sure the correct output shape is returned for the original images
        img_new = np.asarray(img_new[0])[0]
        ref_new = np.asarray(ref_new[0])[0]

        return img_new_training, ref_new_training, img_new, ref_new