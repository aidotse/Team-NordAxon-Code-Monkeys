import os
import random
from time import time
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.unets import UnetResnet152

if __name__ == "__main__":
    
    cfg = {
        "model_params": {
            "class": "UnetResnet152",
        },
        "save_path": "weights",
        "epochs": 400,
        "num_workers": 16,
        "save_checkpoints": True,
        "load_checkpoint": True,#False,

        "train_params": {
            "batch_size": 32,
            "shuffle": True,
        },

        "valid_params": {
            "batch_size": 32,
            "shuffle": False,
        }
    }

    Path("output").mkdir(exist_ok=True, parents=True)

    #train_dataset = AstraZenecaDataset("../data/training_dataset/train", transform=training_safe_augmentations)
    #valid_dataset = AstraZenecaDataset("../data/training_dataset/valid", transform=None)
    
    dataset = ExampleDataset("../data/03_training_data/normalized_bias/valid", transform=test_augmentations(crop_size=(256,256)), test=True)

    dataloader = DataLoader(dataset, batch_size=cfg["train_params"]["batch_size"], num_workers=cfg["num_workers"], shuffle=cfg["train_params"]["shuffle"])

    # TODOS:
    # |x| Save the latest weight and also save the latest weights
    # - What is a scheduler?
    # - Validation will use same metric for all models we will ever train
    # - Train can use different trianing loss functions
    # - What to log
    # |x| Spectral regularizer
    # - Gan training

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = UnetResnet152(output_channels=1)
        
    # Load checkpoints 
    if cfg["load_checkpoint"]:
        weight_file = os.path.join(cfg["save_path"], 'last.pth')
        checkpoint = torch.load(weight_file, map_location=device)
        epoch = checkpoint['epoch']
        best_valid_loss = checkpoint['best_valid_loss']
        model.load_state_dict(checkpoint['model_state_dict'])
            
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    model.to(device)
    print(f"Starting from epoch {epoch}")
        
    model.eval()        
    torch.set_grad_enabled(False)

    with tqdm(total=len(valid_dataset), desc=f'Epoch {epoch + 1}/{cfg["epochs"]}', unit='img') as pbar:
        for inputs in dataloader:
            inputs = inputs.float()

            inputs = inputs.to(device)
            targets = targets[:,0].unsqueeze(1).to(device)            
            
            preds = model(inputs)

            pbar.set_postfix(**{'valid loss: ': np.mean(valid_losses)})
            pbar.update(inputs.shape[0])
    
                        
