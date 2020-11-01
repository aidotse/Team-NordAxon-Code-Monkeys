import numpy as np
import torch
import torch.nn as nn

def SpectralLoss(device, epsilon=1e-8):
    """Spectral regularizer for learning the power curve
    
    Requires images of format
    (b, c, w, h)
    """
    N = 88
    criterion_freq = nn.BCELoss()
    
    def spectral_loss(x_fake, x_real):
        # fake image 1d power spectrum
        psd1D_img = np.zeros([x_fake.shape[0], N])
        for t in range(x_fake.shape[0]):
            gen_imgs = x_fake.permute(0,2,3,1)
            img_numpy = gen_imgs[t,:,:,:].cpu().detach().numpy()
            img_gray = RGB2gray(img_numpy)
            fft = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(fft)
            fshift += epsilon
            magnitude_spectrum = 20*np.log(np.abs(fshift))
            psd1D = azimuthalAverage(magnitude_spectrum)
            psd1D = (psd1D-np.min(psd1D))/(np.max(psd1D)-np.min(psd1D))
            psd1D_img[t,:] = psd1D
        
        psd1D_img = torch.from_numpy(psd1D_img).float()
        psd1D_img = torch.Variable(psd1D_img, requires_grad=True).to(device)
            
        # real image 1d power spectrum
        psd1D_rec = np.zeros([x_real.shape[0], N])
        for t in range(x_real.shape[0]):
            gen_imgs = x_real.permute(0,2,3,1)
            img_numpy = gen_imgs[t,:,:,:].cpu().detach().numpy()
            img_gray = RGB2gray(img_numpy)
            fft = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(fft)
            fshift += epsilon
            magnitude_spectrum = 20*np.log(np.abs(fshift))
            psd1D = azimuthalAverage(magnitude_spectrum)           
            psd1D = (psd1D-np.min(psd1D))/(np.max(psd1D)-np.min(psd1D))
            psd1D_rec[t,:] = psd1D
                
        psd1D_rec = torch.from_numpy(psd1D_rec).float()
        psd1D_rec = torch.Variable(psd1D_rec, requires_grad=True).to(device)

        loss_freq = criterion_freq(psd1D_rec,psd1D_img.detach())
        return loss_freq
    return spectral_loss

# from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
# and https://github.com/cc-hpc-itwm/UpConv/blob/master/Experiments_Codes/radialProfile.py


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


def RGB2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

if __name__ == "__main__":
    freq_criterion = SpectralLoss("cpu", epsilon=1e-3)
    
    freq_criterion(torch.zeros(1, 3, 100, 100), torch.zeros(1, 3, 100, 100))
    