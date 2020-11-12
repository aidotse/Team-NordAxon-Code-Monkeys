import os
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import get_image_metadata
class ExampleDataset(Dataset):

    def __init__(self, dataset_path, crop_size=(256,256), transform=None, test=False):
        """Example dataset for sample images for the Astra Zeneca competition
        
        Group by row_col and field of view
        # row_col
        # field of view
         
        Input and Target share these common values:
        - row_col       = sample id? 
        - field of view = amount of zoom

        For identifying INPUT:
        - action_list_number A04
        - imaging_channel    C04
        - z_number_3d        Z01 - Z07

        For identifying TARGET:
        - action_list_number A01 A02 and A03
        - imaging_channel    C01, C02, C03
        - z_number_3d        Z01
        """
        self.dataset_path = dataset_path
        
        dataset_samples = glob.glob(os.path.join(self.dataset_path, "*/*/Assay*"))

        dataset_dicts = [get_image_metadata(path) for path in dataset_samples]
        
        # Group all 7 inputs with all 3 respective targets into variable sample
        samples = dict()
        for sample_dict in dataset_dicts:
            sample_key = (sample_dict["row_col"], sample_dict["field of view"])

            if samples.get(sample_key) is None:
                samples[sample_key] = {
                    "input": dict(),
                    "target": dict(),
                    "mask" : dict()
                }

            if sample_dict["action_list_number"] == "A04":
                # Is an input
                z_number_3d = sample_dict["z_number_3d"]
                samples[sample_key]["input"][z_number_3d] = sample_dict["path"]
            elif sample_dict["is_mask"]:
                # Is a mask
                action_list_number = sample_dict["action_list_number"]
                samples[sample_key]["mask"][action_list_number] = sample_dict["path"]
            else:
                # Is a target
                action_list_number = sample_dict["action_list_number"]
                samples[sample_key]["target"][action_list_number] = sample_dict["path"]                

        self.samples = list(samples.values())
        self.crop_size = crop_size
        self.transforms = transform
        self.test = test
        
    def __len__(self):
        return len(self.samples) * 32 
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Modulo
        idx = idx % len(self.samples)

        sample_dict = self.samples[idx]
        w, h = cv2.imread(sample_dict["input"]["Z01"], -1).shape
        assert self.crop_size[0] <= w
        assert self.crop_size[1] <= h

        input = np.zeros((7, w, h))
        output = np.zeros((3, w, h))
        mask = np.zeros((3, w, h), dtype = 'int16') # As masks will be binary
        for i, z_number_3d in enumerate(["Z01", "Z02", "Z03", "Z04", "Z05", "Z06", "Z07"]):
            img_path = sample_dict["input"][z_number_3d]
            img = cv2.imread(img_path, -1)
            #img = img.astype(np.int16)
            input[i] = img

        for i, action_list_number in enumerate(["A01", "A02", "A03"]):
            img_path = sample_dict["target"][action_list_number]
            img = cv2.imread(img_path, -1)
            #img = img.astype(np.int16)
            output[i] = img
        
        # add real nuclei mask -- saved as pickle because of problems... ---- /Isabella
        #import pickle
        mask_path = sample_dict["mask"]["A01"]
        #m = pickle.load(open(mask_path, "rb"))
        m = cv2.imread(mask_path, -1)
        # This code should stay:
        mask[0] = m
        
        if self.transforms:
            for transform in self.transforms: 
                input, output, mask = transform(input, output, mask)

        return input, output, mask

class SingleMagnificationDataset(Dataset):

    def __init__(self, dataset_path, magnification, crop_size=(256,256), transform=None, test=False):
        """Example dataset for sample images for the Astra Zeneca competition
        
        Group by row_col and field of view
        # row_col
        # field of view
         
        Input and Target share these common values:
        - row_col       = sample id? 
        - field of view = amount of zoom

        For identifying INPUT:
        - action_list_number A04
        - imaging_channel    C04
        - z_number_3d        Z01 - Z07

        For identifying TARGET:
        - action_list_number A01 A02 and A03
        - imaging_channel    C01, C02, C03
        - z_number_3d        Z01
        """
        self.dataset_path = dataset_path

        dataset_samples = glob.glob(os.path.join(self.dataset_path, "*/*/Assay*"))

        assert magnification in ["20x_images", "40x_images", "60x_images"]
        dataset_dicts = [get_image_metadata(path) for path in dataset_samples if magnification in path]

        # Group all 7 inputs with all 3 respective targets into variable sample
        samples = dict()
        for sample_dict in dataset_dicts:
            sample_key = (sample_dict["row_col"], sample_dict["field of view"])

            if samples.get(sample_key) is None:
                samples[sample_key] = {
                    "input": dict(),
                    "target": dict(),
                    "mask" : dict()
                }

            if sample_dict["action_list_number"] == "A04":
                # Is an input
                z_number_3d = sample_dict["z_number_3d"]
                samples[sample_key]["input"][z_number_3d] = sample_dict["path"]
            elif sample_dict["is_mask"]:
                # Is a mask
                action_list_number = sample_dict["action_list_number"]
                samples[sample_key]["mask"][action_list_number] = sample_dict["path"]
            else:
                # Is a target
                action_list_number = sample_dict["action_list_number"]
                samples[sample_key]["target"][action_list_number] = sample_dict["path"]                

        self.samples = list(samples.values())
        self.crop_size = crop_size
        self.transforms = transform
        self.test = test
        
    def __len__(self):
        return len(self.samples) * 32 
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Modulo
        idx = idx % len(self.samples)

        sample_dict = self.samples[idx]
        w, h = cv2.imread(sample_dict["input"]["Z01"], -1).shape
        assert self.crop_size[0] <= w
        assert self.crop_size[1] <= h

        input = np.zeros((7, w, h))
        output = np.zeros((3, w, h))
        mask = np.zeros((3, w, h), dtype = 'int16') # As masks will be binary
        for i, z_number_3d in enumerate(["Z01", "Z02", "Z03", "Z04", "Z05", "Z06", "Z07"]):
            img_path = sample_dict["input"][z_number_3d]
            img = cv2.imread(img_path, -1)
            #img = img.astype(np.int16)
            input[i] = img

        for i, action_list_number in enumerate(["A01", "A02", "A03"]):
            img_path = sample_dict["target"][action_list_number]
            img = cv2.imread(img_path, -1)
            #img = img.astype(np.int16)
            output[i] = img
        
        # add real nuclei mask -- saved as pickle because of problems... ---- /Isabella
        #import pickle
        mask_path = sample_dict["mask"]["A01"]
        #m = pickle.load(open(mask_path, "rb"))
        m = cv2.imread(mask_path, -1)
        # This code should stay:
        mask[0] = m
        
        if self.transforms:
            for transform in self.transforms: 
                input, output, mask = transform(input, output, mask)

        return input, output, mask

class PredctionDataset(Dataset):
    def __init__(self, dir_path, crop_size=(256,256), transform=None):
        """Prediction dataset for sample images for the Astra Zeneca competition
        
        Group by row_col and field of view
        # row_col
        # field of view
         
        Input and Target share these common values:
        - row_col       = sample id? 
        - field of view = amount of zoom

        For identifying INPUT:
        - action_list_number A04
        - imaging_channel    C04
        - z_number_3d        Z01 - Z07

        For identifying TARGET:
        - action_list_number A01 A02 and A03
        - imaging_channel    C01, C02, C03
        - z_number_3d        Z01
        """
        self.dir_path = dir_path
        
        dataset_samples = glob.glob(os.path.join(self.dir_path, "Assay*"))

        dataset_dicts = [get_image_metadata(path) for path in dataset_samples]
        
        # Group all 7 inputs with all 3 respective targets into variable sample
        samples = dict()
        for sample_dict in dataset_dicts:
            sample_key = (sample_dict["row_col"], sample_dict["field of view"])

            if samples.get(sample_key) is None:
                samples[sample_key] = {
                    "input": dict(),
                    "target": dict(),
                    "mask" : dict()
                }

            if sample_dict["action_list_number"] == "A04":
                # Is an input
                z_number_3d = sample_dict["z_number_3d"]
                samples[sample_key]["input"][z_number_3d] = sample_dict["path"]
            elif sample_dict["is_mask"]:
                # Is a mask
                action_list_number = sample_dict["action_list_number"]
                samples[sample_key]["mask"][action_list_number] = sample_dict["path"]
            else:
                # Is a target
                action_list_number = sample_dict["action_list_number"]
                samples[sample_key]["target"][action_list_number] = sample_dict["path"]                

        self.samples = list(samples.values())
        self.crop_size = crop_size
        self.transforms = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Modulo
        idx = idx % len(self.samples)

        sample_dict = self.samples[idx]
        w, h = cv2.imread(sample_dict["input"]["Z01"], -1).shape
        assert self.crop_size[0] <= w
        assert self.crop_size[1] <= h

        input = np.zeros((7, w, h))
        output = np.zeros((3, w, h))
        mask = np.zeros((3, w, h), dtype = 'int16') # As masks will be binary
        input_filenames = list()
        for i, z_number_3d in enumerate(["Z01", "Z02", "Z03", "Z04", "Z05", "Z06", "Z07"]):
            img_path = sample_dict["input"][z_number_3d]
            img = cv2.imread(img_path, -1)
            input[i] = img
            input_filenames.append(os.path.basename(img_path))
        
        if self.transforms:
            for transform in self.transforms: 
                input = transform(input)
                
        output_filenames = list()
        for target in ["01", "02", "03"]:
            tf = input_filenames[0]
            target_filename = tf[:43] + 'A' + target + tf[46:49] + 'C' + target + tf[52:]
            #target_filename[43:46] = 'A' + target
            #target_filename[49:52] = 'C' + target
            
            output_filenames.append(target_filename)
        return input, input_filenames, output_filenames