import os, argparse
import random
from time import time
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
import wandb

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, default='A2',  help="Either 'A1', 'A2' or 'A3'")
    parser.add_argument('--experimental-augmentations', type=bool, default=False,  help="Use affine- (safe) or experimental (risky) augmentations")
    parser.add_argument('--mask-input', type=bool, default=False,  help="Train with masks as additional input")
    opt = parser.parse_args()
    print(opt)
    
    target_idx = {"A1": 0, "A2": 1, "A3": 2}[opt.target]

    cfg = {
        "model_params": {
            "class": "UnetResnet152v2",
        },
        "save_path": "weights",
        "epochs": 800,
        "num_workers": 16,
        "save_checkpoints": True,
        "load_checkpoint": False,#True,#False,

        "train_params": {
            "batch_size": 32,
            "shuffle": True,
        },

        "valid_params": {
            "batch_size": 32,
            "shuffle": False,
        }
    }
    with wandb.init(project="hackathon-astrazeneca", config=cfg):
        Path(cfg["save_path"]).mkdir(exist_ok=True, parents=True)

        #train_dataset = AstraZenecaDataset("../data/training_dataset/train", transform=training_safe_augmentations)
        #valid_dataset = AstraZenecaDataset("../data/training_dataset/valid", transform=None)
    
        if opt.experimental_augmentations:
            train_dataset = ExampleDataset("../data/03_training_data/normalized_bias/train", transform=all_augmentations())
        else:
            train_dataset = ExampleDataset("../data/03_training_data/normalized_bias/train", transform=affine_augmentations())
        valid_dataset = ExampleDataset("../data/03_training_data/normalized_bias/valid", transform=test_augmentations(crop_size=(256,256)), test=True)

        train_dataloader = DataLoader(train_dataset, batch_size=cfg["train_params"]["batch_size"], num_workers=cfg["num_workers"], shuffle=cfg["train_params"]["shuffle"])
        valid_dataloader = DataLoader(valid_dataset, batch_size=cfg["valid_params"]["batch_size"], num_workers=cfg["num_workers"], shuffle=cfg["valid_params"]["shuffle"])

        # TODO:
        # - What is a scheduler?
        # - Train can use different trianing loss functions
        # - What to log


        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        train_losses = list()
        valid_losses = list()
        
        if opt.mask_input:
            model = UnetResnet152v2(input_channels=8, output_channels=1)
        else:
            model = UnetResnet152v2(input_channels=7, output_channels=1)
        wandb.watch(model, log="all")
        criterion = nn.L1Loss(reduction="mean")
        freq_criterion = SpectralLoss(device)
        optimizer = optim.Adam(model.parameters())
    
        # Load checkpoints 
        if cfg["load_checkpoint"]:
            weight_file = os.path.join(cfg["save_path"], 'last.pth')
            checkpoint = torch.load(weight_file, map_location=device)
            epoch = checkpoint['epoch']
            best_valid_loss = checkpoint['best_valid_loss']
            model.load_state_dict(checkpoint['model_state_dict'])

        else:
            epoch = 0
            best_valid_loss = 1e100
        
        model.to(device)
        print(f"Starting from epoch {epoch}")

        for epoch in range(epoch, cfg["epochs"]):
        
            model.train()
            torch.set_grad_enabled(True)

            train_loss = 0
            valid_loss = 0
        
            with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{cfg["epochs"]}', unit='img') as pbar:
                time_0 = time()
                for inputs, targets, masks in train_dataloader:
                    
                    time_1 = time()
                    inputs, targets, masks = inputs.float(), targets.float(), masks.float()

                    
                    print(masks.shape, inputs.shape, masks[:,target_idx].unsqueeze(1).shape)
                    if opt.mask_input:
                        inputs = torch.cat([
                            inputs, 
                            masks[:,target_idx].unsqueeze(1)
                            ], dim=1)
                        inputs = inputs.to(device)
                    else:
                        inputs = inputs.to(device)
                    
                    targets = targets[:,target_idx].unsqueeze(1).to(device)
                
                    preds = model(inputs)
                    
                    loss = criterion(preds, targets) + freq_criterion(preds, targets)
                
                    train_loss += loss.item() 
                    time_2 = time()

                    optimizer.zero_grad()
                    loss.backward()#nn.utils.clip_grad_value_(model.parameters(), 0.1)
                
                    optimizer.step()

                    train_losses.append(loss.item())

                    pbar.set_postfix(**{'train loss: ': np.mean(train_losses)})
                    pbar.update(inputs.shape[0])
                    time_0 = time()
            model.eval()
            
            torch.set_grad_enabled(False)

            with tqdm(total=len(valid_dataset), desc=f'Epoch {epoch + 1}/{cfg["epochs"]}', unit='img') as pbar:
                for inputs, targets, masks in valid_dataloader:
                    inputs, targets, masks = inputs.float(), targets.float(), masks.float()

                    if opt.mask_input:
                        inputs = torch.cat([
                            inputs, 
                            masks[:,target_idx].unsqueeze(1)
                            ], dim=1)
                        inputs = inputs.to(device)
                    else:
                        inputs = inputs.to(device)
                        
                    targets = targets[:,target_idx].unsqueeze(1).to(device)
                
                    preds = model(inputs)

                    loss = criterion(preds, targets)
                
                    valid_loss += loss.item()

                    valid_losses.append(loss.item())

                    pbar.set_postfix(**{'valid loss: ': np.mean(valid_losses)})
                    pbar.update(inputs.shape[0])

            wandb.log({
                "epoch":epoch, 
                "A3: train MAE": np.mean(train_losses),
                "A3: valid MAE": np.mean(valid_losses),
                "data loading time": time_1 - time_0,
                "forward pass time": time_2 - time_1,
                "backpropagation time": time_0 - time_2,
                })

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
