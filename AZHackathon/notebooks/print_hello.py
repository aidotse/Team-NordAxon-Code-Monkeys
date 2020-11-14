import os, glob, argparse
from pathlib import Path
from tqdm import tqdm
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='testing-pytorch-implementation')
    parser.add_argument('--input_dir', type=str, default='../data/03_training_data/normalized_bias/train/input/20x_images', help='a2b or b2a')
    parser.add_argument('--output_dir', type=str, default="../data/04_generated_images/20x_images", help='input image channels')
    
    opt = parser.parse_args()
    input_dir = opt.input_dir 
    output_dir = opt.output_dir 
    
    print("doing all kinds of advanced stuff..")
    for i in tqdm(range(10)):
        time.sleep(0.5)
    
    
    print("hello world")
    print(input_dir)
    print(output_dir)