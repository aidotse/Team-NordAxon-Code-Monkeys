from __future__ import print_function
import argparse
import os
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from data.dataset import ExampleDataset
from data.augmentations import affine_augmentations, test_augmentations
import models.network as network, utils.gan_util as util
from models.unets import UnetResnet152
from utils.losses import SpectralLoss
# Init wandb
import wandb


from gan.networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
    parser.add_argument('--input_nc', type=int, default=7, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
    opt = parser.parse_args()

    print(opt)

    cfg = {
        "model_params": {
          "class": "UnetResnet152",
        },
        "save_path": "weights",
        "epochs": 800,
        "num_workers": 16,
        "save_checkpoints": True,
        "load_checkpoint": False,#True,#False,

        "train_params": {
            "batch_size": 16,
            "shuffle": True,
        },

        "valid_params": {
            "batch_size": 16,
            "shuffle": False,
        }
    }

    with wandb.init(project="hackathon-astrazeneca", config=cfg):
    
        #cudnn.benchmark = Tru
        print("Loading datasets")
        
        # data_loader
        train_set = ExampleDataset("../data/03_training_data/normalized_bias/train", transform=affine_augmentations())
        test_set = ExampleDataset("../data/03_training_data/normalized_bias/valid", transform=test_augmentations(crop_size=(256,256)))
    
        training_data_loader = DataLoader(train_set, batch_size=cfg["train_params"]["batch_size"], num_workers=cfg["num_workers"], shuffle=cfg["train_params"]["shuffle"])
        testing_data_loader = DataLoader(test_set, batch_size=cfg["valid_params"]["batch_size"], num_workers=cfg["num_workers"], shuffle=cfg["valid_params"]["shuffle"])
    
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
        print('===> Building models')
        net_g = UnetResnet152(input_channels=7, output_channels=1).to(device)
        net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)
        wandb.watch(net_g, log="all")
    
        criterionGAN = GANLoss().to(device)
        criterionL1 = nn.L1Loss().to(device)
        criterionFreq = SpectralLoss(device)

        # setup optimizer
        optimizer_g = optim.Adam(net_g.parameters(), lr=1e-3, betas=(opt.beta1, 0.999))
        optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        net_g_scheduler = get_scheduler(optimizer_g, opt)
        net_d_scheduler = get_scheduler(optimizer_d, opt)


                # Load checkpoints 
        if cfg["load_checkpoint"]:
            g_weight_file = os.path.join(cfg["save_path"], 'g_last.pth')
            g_checkpoint = torch.load(g_weight_file, map_location=device)
            epoch = g_checkpoint['epoch']
            best_valid_loss = g_checkpoint['best_valid_loss']
            net_g.load_state_dict(g_checkpoint['model_state_dict'])

            d_weight_file = os.path.join(cfg["save_path"], 'd_last.pth')
            d_checkpoint = torch.load(d_weight_file, map_location=device)
            net_d.load_state_dict(d_checkpoint['model_state_dict'])
        else:
            epoch = 0
            best_valid_loss = 1e100
        
        print(f"Starting from epoch {epoch}")
    
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            net_g.train()

            d_losses = list()
            g_losses = list()
            train_losses = list()
            valid_losses = list()
            # train
            t_iter = tqdm(enumerate(training_data_loader, 1))
            for iteration, (inputs, targets, masks) in t_iter:
        
                # forward
                real_a, real_b = inputs.float().to(device), targets[:,1].unsqueeze(1).float().to(device)
                fake_b = net_g(real_a)
    
                ######################
                # (1) Update D network
                ######################
    
                optimizer_d.zero_grad()
            
                # train with fake
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = net_d.forward(fake_ab.detach())
                loss_d_fake = criterionGAN(pred_fake, False)
    
                # train with real
                real_ab = torch.cat((real_a, real_b), 1)
                pred_real = net_d.forward(real_ab)
                loss_d_real = criterionGAN(pred_real, True)
                
                # Combined D loss
                loss_d = (loss_d_fake + loss_d_real) * 0.5
    
                loss_d.backward()
            
                optimizer_d.step()
    
                ######################
                # (2) Update G network
                ######################
        
                optimizer_g.zero_grad()
        
                # First, G(A) should fake the discriminator
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = net_d.forward(fake_ab)
                loss_g_gan = criterionGAN(pred_fake, True)
        
                # Second, G(A) = B
                loss_g_l1 = criterionL1(fake_b, real_b)
                loss_g_spectral = criterionFreq(fake_b, real_b)

                loss_g = loss_g_gan + loss_g_l1 * opt.lamb + loss_g_spectral
                
                loss_g.backward()
        
                optimizer_g.step()
        
                t_iter.set_description(f"Epoch[{epoch}]: Loss_D: {round(float(loss_d.item()),5)} Loss_G: {round(float(loss_g.item()),5)}")
                train_losses.append(loss_g_l1.item())
                d_losses.append(loss_d.item())
                g_losses.append(loss_g.item())
            update_learning_rate(net_g_scheduler, optimizer_g)
            update_learning_rate(net_d_scheduler, optimizer_d)
        
            # test
            net_g.eval()
            with torch.no_grad():
                t_iter = tqdm(enumerate(testing_data_loader, 1))
                for iteration, (inputs, targets, masks) in t_iter:
            
                    inputs, targets = inputs.float().to(device), targets[:,1].unsqueeze(1).float().to(device)
            
                    prediction = net_g(inputs)
            
                    loss = criterionL1(prediction, targets)
                    valid_losses.append(loss.item())
                    t_iter.set_description(f"Epoch[{epoch}]: Valid Loss_G {np.mean(valid_losses)}")
            wandb.log({
                "epoch": epoch,
                "d_loss": np.mean(d_losses),
                "g_loss": np.mean(g_losses),
                "A2: valid MAE": np.mean(valid_losses),
                "A2: train MAE": np.mean(train_losses)
            })

            # If validation score improves, save the weights
            if best_valid_loss > np.mean(valid_losses):
                best_valid_loss = np.mean(valid_losses)
                
                torch.save({
                    'epoch': epoch + 1,
                    'best_valid_loss': best_valid_loss,
                    'model_state_dict': net_g.state_dict(),
                    os.path.join(cfg["save_path"], 'g_best.pth')
                })  
                
                torch.save({
                    'epoch': epoch + 1,
                    'best_valid_loss': best_valid_loss,
                    'model_state_dict': net_d.state_dict(),
                    os.path.join(cfg["save_path"], 'd_best.pth')
                }) 


            # Save latest weights as checkpoints
            if cfg["save_checkpoints"]:
                pass
                """
                torch.save({
                    'epoch': epoch + 1,
                    'best_valid_loss': best_valid_loss,
                    'model_state_dict': net_g.state_dict(),
                    os.path.join(cfg["save_path"], 'g_last.pth')
                    })
                torch.save({
                    'epoch': epoch + 1,
                    'best_valid_loss': best_valid_loss,
                    'model_state_dict': net_d.state_dict(),
                    os.path.join(cfg["save_path"], 'd_last.pth')
                    })"""
