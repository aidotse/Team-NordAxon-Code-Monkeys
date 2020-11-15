import os, glob, argparse
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F2

from utils.utils import get_image_metadata
from utils.postprocessing import matching_histograms
from models.unets import UnetResnet152, UnetSegmentationResnet152
from data.dataset import PredictionDataset



def strided_predict(original_inputs:torch.Tensor, 
                    model:torch.nn.Module, 
                    device:torch.device, 
                    crop_size:int = 256, 
                    stride:int = 256, 
                    output_channels:int = 1, 
                    batch_size:int = 16
    ) -> torch.Tensor:
    assert crop_size >= stride # Crop size must be larger than stride
    
    sizes = np.array([2**i for i in range(10)])
    b, c, w, h = original_inputs.shape
    output_image = torch.zeros(b,output_channels,w,h)
    output_counts = torch.zeros(w,h)

    with torch.no_grad():
        model.eval()
        model.to(device)
        
        batch_tensors = list()
        batch_metadata = list()
        
        for i in range(0, w-crop_size+stride, stride):
            
            for j in range(0, h-crop_size+stride, stride):

                x = original_inputs[:,:,i:i+crop_size,j:j+crop_size]

                # Resize rectangular tensors into allowed rectangular tensors
                crop_shape = x.shape
                
                rectangle_flag = False
                if x.shape[2] != crop_size:
                    rectangle_flag = True
                    idx = np.argmin(np.abs(sizes - x.shape[2]))
                    x = F.interpolate(x, size=(sizes[idx], x.shape[3]))
                if x.shape[3] != crop_size:
                    rectangle_flag = True
                    idx = np.argmin(np.abs(sizes - x.shape[3]))
                    x = F.interpolate(x, size=(x.shape[2], sizes[idx]))
                    
                out = model(x.to(device)).detach().cpu()
                out = F.interpolate(out, size=(crop_shape[2], crop_shape[3]))
                output_counts[i:i+crop_size,j:j+crop_size] += torch.ones(crop_shape[2],crop_shape[3])
                output_image[:,:,i:i+crop_size,j:j+crop_size] += out
                
                #TODO: Implement inference in batches for speed-up (see notebook)

    output_image = output_image * (1 / output_counts)
    return output_image

def test_time_augmentation_predict(inputs:torch.Tensor, model:nn.Module, device:torch.device, 
                                   crop_size:int = 256, 
                                   stride:int = 256, 
                                   batched:bool =True
    ):
    """
    # 0 - x
    # 1 - F.hflip(x)
    # 2 - F.vflip(x)
    # 3 - F.hflip(F.vflip(x))
    # 4 - F.rotate(x, 90)
    # 5 - F.rotate(F.hflip(x), 90)
    # 6 - F.rotate(F.vflip(x), 90)
    # 7 - F.rotate(F.hflip(F.vflip(x)), 90)
    """
    # Augment (~1 seconds)
    inputs_0 = inputs
    inputs_1 = F2.hflip(inputs_0)
    inputs_2 = F2.vflip(inputs_0)
    inputs_3 = F2.hflip(inputs_2)
    inputs_4 = F2.rotate(inputs_0, 90, expand=True)
    inputs_5 = F2.hflip(inputs_4)
    inputs_6 = F2.vflip(inputs_4)
    inputs_7 = F2.hflip(inputs_6)
    
    # Inference (~100 seconds GPU)
    if batched:
    
        batched_inputs_0 = torch.cat([inputs_0, inputs_1, inputs_2, inputs_3], 0)
        batched_inputs_1 = torch.cat([inputs_4, inputs_5, inputs_6, inputs_7], 0)
        batched_outputs_0 = strided_predict(batched_inputs_0, model, device, crop_size=crop_size, stride=stride)
        batched_outputs_1 = strided_predict(batched_inputs_1, model, device, crop_size=crop_size, stride=stride)
        outputs_0 = batched_outputs_0[0].unsqueeze(0)
        outputs_1 = batched_outputs_0[1].unsqueeze(0)
        outputs_2 = batched_outputs_0[2].unsqueeze(0)
        outputs_3 = batched_outputs_0[3].unsqueeze(0)
        outputs_4 = batched_outputs_1[0].unsqueeze(0)
        outputs_5 = batched_outputs_1[1].unsqueeze(0)
        outputs_6 = batched_outputs_1[2].unsqueeze(0)
        outputs_7 = batched_outputs_1[3].unsqueeze(0)
                    
    else:
        outputs_0 = strided_predict(inputs_0, model, device, crop_size=crop_size, stride=stride)
        outputs_1 = strided_predict(inputs_1, model, device, crop_size=crop_size, stride=stride)
        outputs_2 = strided_predict(inputs_2, model, device, crop_size=crop_size, stride=stride)
        outputs_3 = strided_predict(inputs_3, model, device, crop_size=crop_size, stride=stride)
        outputs_4 = strided_predict(inputs_4, model, device, crop_size=crop_size, stride=stride)
        outputs_5 = strided_predict(inputs_5, model, device, crop_size=crop_size, stride=stride)
        outputs_6 = strided_predict(inputs_6, model, device, crop_size=crop_size, stride=stride)
        outputs_7 = strided_predict(inputs_7, model, device, crop_size=crop_size, stride=stride)
    
    # Revert augmentation on predictions (~1 seconds)
    outputs_1 = F2.hflip(outputs_1)
    outputs_2 = F2.vflip(outputs_2)
    outputs_3 = F2.vflip(F2.hflip(outputs_3))
    outputs_4 = F2.rotate(outputs_4, -90, expand=True)
    outputs_5 = F2.rotate(F2.hflip(outputs_5), -90, expand=True)
    outputs_6 = F2.rotate(F2.vflip(outputs_6), -90, expand=True)
    outputs_7 = F2.rotate(F2.vflip(F2.hflip(outputs_7)), -90, expand=True)
    
    outputs = outputs_0 + outputs_1 + outputs_2 + outputs_3 + outputs_4 + outputs_5 + outputs_6 + outputs_7
    return outputs / 8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    parser.add_argument('--input-dir', type=str, default='../data/03_training_data/normalized_bias/train/input/20x_images', help='a2b or b2a')
    parser.add_argument('--output-dir', type=str, default="../data/04_generated_images/20x_images", help='input image channels')
    parser.add_argument('--weights_path', type=str, default="../../data/05_saved_models/normalized_bias/A1_segmentation_model_2.pth", help='output image channels')
    parser.add_argument('--stride', type=int, default=512, help="(Must be less than 'crop-size') Must be 2^n, e.g. 256, 512, 1024")
    parser.add_argument('--crop-size', type=int, default=1024, help="Must be 2^n, e.g. 256, 512, 1024")

    opt = parser.parse_args()
    print(opt)

    # 1 Load model and weights
    # 2 Create prediction dataset
    
    # 4 Save images (and display plot + metrics)
    
    input_dir = opt.input_dir 
    output_dir = opt.output_dir 
    weights_path = opt.weights_path 
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    dataset = PredictionDataset(input_dir, use_masks=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if opt.mask:
        model = UnetSegmentationResnet152(input_channels=7, output_channels=1)
    else:
        model = UnetResnet152(input_channels=7, output_channels=1)
        

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
   
    dataset_length = len(dataset)
    for i in tqdm(range(dataset_length)):
        
        inputs, input_filenames, output_filenames = dataset[i]
            
        x = torch.Tensor(inputs).unsqueeze(0)
        output_image = test_time_augmentation_predict(x, model, device, 
                                                      crop_size=opt.crop_size, 
                                                      stride=opt.stride)
           
        generated_image = (output_image[0,0]>0.5).numpy().astype(np.uint8)  
        
        save_path = os.path.join(output_dir, output_filenames[0])
        success = cv2.imwrite(save_path, generated_image)
        if success:
            print(f"Successfully saved to {save_path}")
        else:
            print(f"Failed to save {save_path}")
