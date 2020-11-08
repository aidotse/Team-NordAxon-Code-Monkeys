import os, time, pickle, argparse

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
#from torch.autograd import Variable

from data.dataset import ExampleDataset
from data.augmentations import affine_augmentations, test_augmentations
import models.network as network, utils.gan_util as util
from models.unets import UnetResnet152

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='facades',  help='')
    parser.add_argument('--train_subfolder', required=False, default='train',  help='')
    parser.add_argument('--test_subfolder', required=False, default='val',  help='')
    parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
    parser.add_argument('--test_batch_size', type=int, default=5, help='test batch size')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--input_size', type=int, default=256, help='input size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
    parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
    parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True or False')
    parser.add_argument('--train_epoch', type=int, default=200, help='number of train epochs')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--L1_lambda', type=float, default=100, help='lambda for L1 loss')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--save_root', required=False, default='results', help='results save path')
    parser.add_argument('--inverse_order', type=bool, default=True, help='0: [input, target], 1 - [target, input]')
    opt = parser.parse_args()
    print(opt)

    # results save path
    root = opt.dataset + '_' + opt.save_root + '/'
    model = opt.dataset + '_'
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(root + 'Fixed_results'):
        os.mkdir(root + 'Fixed_results')

    # data_loader
    train_dataset = ExampleDataset("../data/03_training_data/normalized_bias/train", transform=affine_augmentations())
    valid_dataset = ExampleDataset("../data/03_training_data/normalized_bias/valid", transform=test_augmentations())

    train_dataloader = DataLoader(train_dataset)
    valid_dataloader = DataLoader(valid_dataset)

    """transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_loader = util.data_load('data/' + opt.dataset, opt.train_subfolder, transform, opt.batch_size, shuffle=True)
    test_loader = util.data_load('data/' + opt.dataset, opt.test_subfolder, transform, opt.test_batch_size, shuffle=True)
    
    test = valid_dataloader.__iter__().__next__()[0]
    img_size = test.size()[2]
    if opt.inverse_order:
        fixed_y_ = test[:, :, :, 0:img_size]
        fixed_x_ = test[:, :, :, img_size:]
    else:
        fixed_x_ = test[:, :, :, 0:img_size]
        fixed_y_ = test[:, :, :, img_size:]

    if img_size != opt.input_size:
        fixed_x_ = util.imgs_resize(fixed_x_, opt.input_size)
        fixed_y_ = util.imgs_resize(fixed_y_, opt.input_size)
    """

    # GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # network
    G = UnetResnet152(input_channels=7, output_channels=1) #network.generator(opt.ngf)
    D = network.discriminator(opt.ndf)
    #G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.to(device)
    D.to(device)
    G.train()
    D.train()

    # Loss
    BCE_loss = nn.BCELoss().to(device)
    L1_loss = nn.L1Loss().to(device)

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
    D_optimizer = optim.Adam(D.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    print('training start!')
    start_time = time.time()
    for epoch in range(opt.train_epoch):
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()
        num_iter = 0

        t_iter = tqdm(train_dataloader)
        for inputs, targets, masks in t_iter:
            # train discriminator D
            D.zero_grad()

            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

            x_ = inputs
            y_ = targets
            """if opt.inverse_order:
                y_ = x_[:, :, :, 0:img_size]
                x_ = x_[:, :, :, img_size:]
            else:
                y_ = x_[:, :, :, img_size:]
                x_ = x_[:, :, :, 0:img_size]
                
            if img_size != opt.input_size:
                x_ = util.imgs_resize(x_, opt.input_size)
                y_ = util.imgs_resize(y_, opt.input_size)

            if opt.resize_scale:
                x_ = util.imgs_resize(x_, opt.resize_scale)
                y_ = util.imgs_resize(y_, opt.resize_scale)

            if opt.crop_size:
                x_, y_ = util.random_crop(x_, y_, opt.crop_size)

            if opt.fliplr:
                x_, y_ = util.random_fliplr(x_, y_)

            x_, y_ = x_.cuda(), y_.cuda()"""

            D_result = D(x_, y_).squeeze()
            D_real_loss = BCE_loss(D_result, torch.ones(D_result.size()).to(device))

            G_result = G(x_)
            D_result = D(x_, G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, torch.zeros(D_result.size()).to(device))

            D_train_loss = (D_real_loss + D_fake_loss) * 0.5
            D_train_loss.backward()
            D_optimizer.step()

            train_hist['D_losses'].append(D_train_loss.item())

            D_losses.append(D_train_loss.item())

            # train generator G
            G.zero_grad()

            G_result = G(x_)
            D_result = D(x_, G_result).squeeze()

            G_train_loss = BCE_loss(D_result, torch.ones(D_result.size()).cuda()) + opt.L1_lambda * L1_loss(G_result, y_)
            G_train_loss.backward()
            G_optimizer.step()

            train_hist['G_losses'].append(G_train_loss.item())

            G_losses.append(G_train_loss.item())

            num_iter += 1
            t_iter.set_description(f"Epoch {epoch + 1}, Discriminator loss: {np.mean(D_losses)}, Generator loss: {np.mean(G_losses)}")

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), opt.train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                np.mean(G_losses)))
        fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
        util.show_result(G, fixed_x_.to(device), fixed_y_, (epoch+1), save=True, path=fixed_p)
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.train_epoch, total_ptime))
    print("Training finish!... save training results")
    torch.save(G.state_dict(), root + model + 'generator_param.pkl')
    torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
    with open(root + model + 'train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    util.show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')
    util.generate_animation(root, model, opt)
