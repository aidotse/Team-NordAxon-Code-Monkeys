import os
import random
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.augmentations import affine_augmentations, test_augmentations
from data.dataset import ExampleDataset
from models.unets import UnetResnet152
from utils.losses import SpectralLoss


# Init wandb
#import wandb
#wandb.init(project='astrazeneca')
# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

if __name__ == "__main__":
    
    cfg = {
        "model_params": {
            "class": "UnetResnet152",
        },
        "save_path": "weights",
        "epochs": 400,
        "num_workers": 0,
        "save_checkpoints": True,
        "load_checkpoint": False,

        "train_params": {
            "batch_size": 64,
            "shuffle": True,
        },

        "valid_params": {
            "batch_size": 1,
            "shuffle": False,
        }
    }
    Path(cfg["save_path"]).mkdir(exist_ok=True, parents=True)

    #train_dataset = AstraZenecaDataset("../data/training_dataset/train", transform=training_safe_augmentations)
    #valid_dataset = AstraZenecaDataset("../data/training_dataset/valid", transform=None)
    
    train_dataset = ExampleDataset("../data/03_training_data/train", transform=affine_augmentations())
    valid_dataset = ExampleDataset("../data/03_training_data/valid", transform=test_augmentations(), test=True)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["train_params"]["batch_size"], num_workers=cfg["num_workers"], shuffle=cfg["train_params"]["shuffle"])
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg["valid_params"]["batch_size"], num_workers=cfg["num_workers"], shuffle=cfg["valid_params"]["shuffle"])

    # TODOS:
    # |x| Save the latest weight and also save the latest weights
    # - What is a scheduler?
    # - Validation will use same metric for all models we will ever train
    # - Train can use different trianing loss functions
    # - What to log
    # |x| Spectral regularizer
    # - Gan training

    save_cp = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #device = "cpu"

    train_losses = list()
    valid_losses = list()

    model = UnetResnet152()

    criterion = nn.L1Loss()
    freq_criterion = SpectralLoss(device)
    optimizer = optim.Adam(model.parameters())
    
    # Load checkpoints 
    if cfg["load_checkpoint"]:
        weight_file = os.path.join(cfg["save_path"], 'last.pth')
        checkpoint = torch.load(weight_file, map_location=device)
        epoch = checkpoint['epoch']
        best_valid_loss = checkpoint['best_valid_loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        epoch = 0
        best_valid_loss = 1e100
    print(f"Starting from epoch {epoch}")

    model.to(device)
    for epoch in range(epoch, cfg["epochs"]):
        
        model.train()
        torch.set_grad_enabled(True)

        train_loss = 0
        valid_loss = 0
        
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{cfg["epochs"]}', unit='img') as pbar:
            for inputs, targets, masks in train_dataloader:
    
                inputs, targets, masks = inputs.float(), targets.float(), masks.float()

                inputs = inputs.to(device)
                targets = targets.to(device)
                
                preds = model(inputs)

                loss = criterion(preds, targets) + freq_criterion(preds, targets)
                
                train_loss += loss.item() 

                optimizer.zero_grad()
                loss.backward()#nn.utils.clip_grad_value_(model.parameters(), 0.1)
                
                optimizer.step()

                train_losses.append(loss.item())

                pbar.set_postfix(**{'train loss: ': np.mean(train_losses)})
                pbar.update(inputs.shape[0])

        model.eval()
        torch.set_grad_enabled(False)

        with tqdm(total=len(valid_dataset), desc=f'Epoch {epoch + 1}/{cfg["epochs"]}', unit='img') as pbar:
            for inputs, targets, masks in valid_dataloader:
                inputs, targets, masks = inputs.float(), targets.float(), masks.float()

                inputs = inputs.to(device)
                targets = targets.to(device)
                
                preds = model(inputs)

                loss = criterion(preds, targets)
                
                valid_loss += loss.item()

                valid_losses.append(loss.item())

                pbar.set_postfix(**{'valid loss: ': np.mean(valid_losses)})
                pbar.update(inputs.shape[0])
        
        # If validation score improves, save the weights
        if best_valid_loss > np.mean(valid_losses):
            best_valid_loss = np.mean(valid_losses)
            torch.save({
                'epoch': epoch + 1,
                'best_valid_loss': best_valid_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(cfg["save_path"], 'best.pth')
                )     

        # Save latest weights as checkpoints
        if cfg["save_checkpoints"]:
            torch.save({
                'epoch': epoch + 1,
                'best_valid_loss': best_valid_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(cfg["save_path"], 'last.pth')
                )     












    torch.save(net.state_dict(), f'last{epoch + 1}.pth')
