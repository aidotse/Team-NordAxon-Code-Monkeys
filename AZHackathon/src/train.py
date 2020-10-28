
import torch
from torch.data import DataLoader


def

if __name__ == "__main__":
    
    cfg = {
        "model_params": {
            "class": UNetResnet152,
        },
        "save_path": "weights",
        "epochs": 40,
        "num_workers": 8,

        "train_params": {
            "batch_size": 32,
            "shuffle": True,
        }

        "valid_params": {
            "batch_size": 64,
            "shuffle": False,
        }
    }

    train_dataset = AstraZenecaDataset("../data/training_dataset/train", transform=training_safe_augmentations)
    valid_dataset = AstraZenecaDataset("../data/training_dataset/valid", transform=None)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], num_workers=cfg["num_workers"], shuffle=cfg["train"]["shuffle"])
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg["valid"]["batch_size"], num_workers=cfg["num_workers"], shuffle=cfg["valid"]["shuffle"])

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

    model = cfg["model_params"]["model_params"]()
    model.to(device)

    criterion =
    optimizer = 
    
    for epoch in range(cfg["epochs"]):
        
        model.train()
        torch.set_grad_enabled(True)

        train_loss = 0
        valid_loss = 0

        with tqdm(total=len(dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for inputs, targets in dataloader:

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

                pbar.set_postfix(**{'train loss: ': np.mean(train_losses))
                pbar.update(inputs.shape[0])

        model.eval()
        torch.set_grad_enabled(False)

        with tqdm(total=len(validation_dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
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
