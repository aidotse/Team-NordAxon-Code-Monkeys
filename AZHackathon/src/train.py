
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.augmentations import affine_augmentations
from data.dataset import ExampleDataset
from models.unets import UnetResnet152


if __name__ == "__main__":
    
    cfg = {
        "model_params": {
            "class": "UnetResnet152",
        },
        "save_path": "weights",
        "epochs": 40,
        "num_workers": 0,

        "train_params": {
            "batch_size": 32,
            "shuffle": True,
        },

        "valid_params": {
            "batch_size": 64,
            "shuffle": False,
        }
    }

    #train_dataset = AstraZenecaDataset("../data/training_dataset/train", transform=training_safe_augmentations)
    #valid_dataset = AstraZenecaDataset("../data/training_dataset/valid", transform=None)

    train_dataset = ExampleDataset("../data/03_train_valid", transform=affine_augmentations())
    valid_dataset = ExampleDataset("../data/03_train_valid", transform=None)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["train_params"]["batch_size"], num_workers=cfg["num_workers"], shuffle=cfg["train_params"]["shuffle"])
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg["valid_params"]["batch_size"], num_workers=cfg["num_workers"], shuffle=cfg["valid_params"]["shuffle"])

    # TODOS:
    # - Save the latest weight and also save the latest weights
    # - What is a scheduler?
    # - Validation will use same metric for all models we will ever train
    # - Train can use different trianing loss functions
    # - What to logg
    # - Spectral regularizer
    # - Gan training

    global_step = 0
    save_cp = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_losses = 0
    valid_losses = 0

    model = UnetResnet152()
    model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(cfg["epochs"]):
        
        model.train()
        torch.set_grad_enabled(True)

        train_loss = 0
        valid_loss = 0
        print("Tast")
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{cfg["epochs"]}', unit='img') as pbar:
            for inputs, targets in train_dataloader:
                print("Test")

                inputs = inputs.to(device)
                targets = targets.to(device)
                
                preds = model(inputs)

                loss = criterion(preds, targets)
                
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                #nn.utils.clip_grad_value_(model.parameters(), 0.1)
                
                optimizer.step()

                train_losses.append(loss.item())

                pbar.set_postfix(**{'train loss: ': np.mean(train_losses)})
                pbar.update(inputs.shape[0])

        model.eval()
        torch.set_grad_enabled(False)

        with tqdm(total=len(validation_dataset), desc=f'Epoch {epoch + 1}/{cfg["epochs"]}', unit='img') as pbar:
            for inputs, targets in validation_dataloader:

                inputs = inputs.to(device)
                targets = targets.to(device)
                
                preds = model(inputs)

                loss = criterion(preds, targets)
                
                valid_loss += loss.item()

                valid_losses.append(loss.item())

                pbar.set_postfix(**{'valid loss: ': np.mean(valid_losses)})
                pbar.update(inputs.shape[0])

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                    dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
