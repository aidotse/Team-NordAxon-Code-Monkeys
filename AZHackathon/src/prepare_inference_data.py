import os
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from utils.utils import get_image_metadata

def create_inference_dataset(input_dir: str, output_dir: str) -> None:
    """Takes Astra Zeneca Adipocyte brightfield images and pre-processed data
    for inference with models of Group 6.
    
    Assumptions made:
        - Data is grouped by magnification [20x_images, 40x_images, 60x_images]
        - Brightfield .tif images have filenames 'AssayPlate_Greiner*' 
        - This script uses 'A04' in the filename to find the input images
    """
    
    stats = {
        "20x_images": (2594.456945, 966.056444), # (mean, std)
        "40x_images": (713.360256, 282.067244),
        "60x_images": (524.969571, 228.204624)
    }
    
    # Get all .tif images
    paths = list()
    for root, dirs, files in os.walk(input_dir, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            if ".tif" in path and "AssayPlate_Greiner" in path:
                paths.append(path)
    print(f"Found {len(paths)} .tif images in total")
    
    inputs = ['A04']
    targets = ['A01', 'A02', 'A03']

    for img_path in tqdm(paths):
        img_metadata = get_image_metadata(img_path)

        is_input = None
        if img_metadata["action_list_number"] in inputs:
            is_input = True
        elif img_metadata["action_list_number"] in targets:
            is_input = False
        else:
            assert False # Code should never reach here unless something is wrong
       
        filename = os.path.basename(img_metadata["path"])
        if is_input:
            magnification = img_metadata["magnification"]
            
            dir_path = os.path.join(output_dir, "input", magnification)
            save_path = os.path.join(dir_path, filename)
            source_path = img_metadata["path"]    
            
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            img = cv2.imread(source_path, -1)
            
            mean = stats[magnification][0]
            std = stats[magnification][1]
            img = (img - mean)/std
            success = cv2.imwrite(save_path, img)
            
            if not success:
                print(f"Failed to save '{save_path}'")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    parser.add_argument('--input-dir', type=str, required=True, help='path to Astra Zeneca Adipocyte dataset (raw)')
    parser.add_argument('--output-dir', type=str, required=True, help="path to where inference data will be saved to")
    opt = parser.parse_args()

    print(opt)
    
    create_inference_dataset(opt.input_dir, opt.output_dir)
    # 1. Get list of all 20x-, 40x- and 60x- magnification input images
    # 2. Generate masks for A1 
    # 3. Predict for A1, A2 and A3
