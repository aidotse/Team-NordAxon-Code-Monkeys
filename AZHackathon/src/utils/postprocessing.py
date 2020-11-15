import torch
from skimage.exposure import match_histograms
import numpy as np 
import cv2
import pickle

def matching_histograms(gen_img : torch.Tensor, mag : str, target : str) -> torch.Tensor:
        try:
            gen_img_np = gen_img.numpy()
            print(gen_img_np.shape)
        except:
            gen_img_np = gen_img

        with open('../../data/06_outputs/target_counters/counter_objects_' + 
                  mag + '_' + target+ '.pickle', 'rb') as handle:
            counter = pickle.load(handle)
        reference_img = []
        for key, val in counter.items():
            reference_img.extend([key for i in range(val)])
        reference_img = np.reshape(np.asarray(reference_img), gen_img_np.shape)
        matched = match_histograms(gen_img_np, reference_img)
        

        return matched
