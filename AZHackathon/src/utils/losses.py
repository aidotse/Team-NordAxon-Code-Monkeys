import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def reverse_huber_loss(pred, true):
    batch_size = pred.shape[0]

    pred = pred.contiguous().view(batch_size, -1)
    true = true.contiguous().view(batch_size, -1)
    loss = (pred - true).abs()
    loss[loss > 1] = loss[loss > 1]**2 
    return loss.mean()

# From https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def SpectralLoss(device: torch.device, epsilon: float=1e-8):
    """Spectral regularizer for learning the power curve
    
    Requires images of format
    (b, c, w, h)
    """
    N = 179
    criterion_freq = nn.BCELoss()
    
    def spectral_loss(x_fake, x_real):
        # fake image 1d power spectrum
        psd1D_img = compute_power_curve(x_fake, epsilon)

        psd1D_img = torch.from_numpy(psd1D_img).float()
        psd1D_img = psd1D_img.to(device)
        psd1D_img.requires_grad = True
            
        # real image 1d power spectrum
        psd1D_rec = compute_power_curve(x_real, epsilon)
            
        psd1D_rec = torch.from_numpy(psd1D_rec).float()
        psd1D_rec = psd1D_rec.to(device)
        psd1D_rec.requires_grad = True

        loss_freq = criterion_freq(psd1D_rec, psd1D_img.detach())
        return loss_freq
    return spectral_loss

# from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
# and https://github.com/cc-hpc-itwm/UpConv/blob/master/Experiments_Codes/radialProfile.py

def compute_power_curve(x: np.array, epsilon: float) -> np.array:
    psd1D_batch = None
    for i in range(x.shape[0]):
        gen_imgs = x.permute(0,2,3,1)
        img_numpy = gen_imgs[i,:,:,:].cpu().detach().numpy()

        img_gray = img_numpy.mean(axis=-1)

        fft = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(fft)
        fshift += epsilon
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        psd1D = azimuthalAverage(magnitude_spectrum)           
        psd1D = (psd1D-np.min(psd1D))/(np.max(psd1D)-np.min(psd1D))

        if psd1D_batch is None:
            psd1D_batch = np.zeros([x.shape[0], len(psd1D)])

        psd1D_batch[i,:] = psd1D
        return psd1D_batch

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


if __name__ == "__main__":
    freq_criterion = SpectralLoss("cpu", epsilon=1e-3)
    
    loss = freq_criterion(torch.ones(16, 2, 256, 256), torch.ones(16, 2, 256, 256))
    print(loss)
