from __future__ import print_function
import argparse
import os

import numpy as numpy
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

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    #parser.add_argument('--dataset', required=True, help='facades')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
    parser.add_argument('--input_nc', type=int, default=7, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
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
        if opt.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
    
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
        optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        net_g_scheduler = get_scheduler(optimizer_g, opt)
        net_d_scheduler = get_scheduler(optimizer_d, opt)
    
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            train_losses = list()
            valid_losses = list()
            # train
            for iteration, (inputs, targets, masks) in enumerate(training_data_loader, 1):
        
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
                loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb + criterionFreq(fake_b, real_b)
                
                loss_g = loss_g_gan + loss_g_l1
                
                loss_g.backward()
        
                optimizer_g.step()
        
                print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                    epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))
                train_losses.append(loss_g_l1.item())
            update_learning_rate(net_g_scheduler, optimizer_g)
            update_learning_rate(net_d_scheduler, optimizer_d)
        
            # test
            for iteration, (inputs, targets, masks) in enumerate(testing_data_loader, 1):
        
                inputs, targets = inputs.float().to(device), targets[:,1].unsqueeze(1).float().to(device)
        
                prediction = net_g(inputs)
        
                mae = criterionL1(prediction, targets)
                valid_losses.append(mae)
            
            wandb.log({
                "epoch": epoch,
                "A2: valid MAE": np.mean(valid_losses),
                "A2: valid MAE": np.mean(train_losses)
            })
        
            #checkpoint
            if epoch % 50 == 0:
                if not os.path.exists("checkpoint"):
                    os.mkdir("checkpoint")
                if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
                    os.mkdir(os.path.join("checkpoint", opt.dataset))
                net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
                net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
                torch.save(net_g, net_g_model_out_path)
                torch.save(net_d, net_d_model_out_path)
                print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))
