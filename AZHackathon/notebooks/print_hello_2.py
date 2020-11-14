import os, glob, argparse
from pathlib import Path
from tqdm import tqdm
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    parser.add_argument('--input_dir', type=str, default='../data/03_training_data/normalized_bias/train/input/20x_images', help='a2b or b2a')
    parser.add_argument('--output_dir', type=str, default="../data/04_generated_images/20x_images", help='input image channels')
    parser.add_argument('--weights_path', type=str, default="../../data/05_saved_models/A2_g_best.pth", help='output image channels')
    parser.add_argument('--target', type=str, default="A2", help="'A1', 'A2', 'A3' or 'all'")
    parser.add_argument('--mask', action='store_true')

    opt = parser.parse_args()
    
    input_dir = opt.input_dir 
    output_dir = opt.output_dir 
    weights_path = opt.weights_path 
    target = opt.target 
    
    print("Predicting images for target " + target +"...")
    for i in tqdm(range(10)):
        time.sleep(0.5)