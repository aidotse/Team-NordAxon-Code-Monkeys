import random
from typing import Tuple

import albumentations as A
import numpy as np

# The cell below shows a wrapper function for most of the augmentations, 
# to simplify the code

def stack(augmentation):
    def wrapper(self, input: list, target: list, mask: list) -> tuple:
        # Pack targets and inputs
        stacked_inputs = np.append(input, target, axis = 2)
        # Transform
        augs = augmentation(self, stacked_inputs, mask)
        stacked_inputs, mask = augs['image'], augs['mask']
        # Unpack
        input = stacked_inputs[:,:,:7]
        target = stacked_inputs[:,:,7:]
        return input, target, mask
    return wrapper

# ----------------- Affine transforms ----------------------------------
#

class VerticalFlip:
    """
    Will flip the inputs vertically with
    a probability P=0.5
    """
    def __init__(self, p = 0.5):
        self.transform = A.VerticalFlip(p=p)

    @stack
    def __call__(self, inputs: list, mask: list) -> dict:
        return self.transform(image = inputs, mask = mask)

class HorizontalFlip:
    """
    Will flip the inputs horizontally with
    a probability P=0.5
    """
    def __init__(self, p = 0.5):
        self.transform = A.HorizontalFlip(p=p)

    @stack
    def __call__(self, inputs: list, mask: list) -> dict:
        return self.transform(image = inputs, mask = mask)

class Rotate:
    """
    Will rotate the inputs between angles +-limit with
    a probability P=0.5
    """
    def __init__(self, p = 0.5, limit = 9):
        self.transform = A.Rotate(limit = limit, p=p)

    @stack
    def __call__(self, inputs: list, mask: list) -> dict:
        return self.transform(image = inputs, mask = mask)

class Transpose:
    """
    Will transpose the inputs with
    a probability P=0.5
    """
    def __init__(self, p = 0.5):
        self.transform = A.Transpose(p=p)

    @stack
    def __call__(self, inputs: list, mask: list) -> dict:
        return self.transform(image = inputs, mask = mask)

class Scale:

    def __init__(self, size: tuple, p = 0.5):
        h, w = size[0], size[1]
        self.transform = A.RandomResizedCrop(h, w, p=p)

    @stack
    def __call__(self, inputs: list, mask: list) -> dict:
        return self.transform(image = inputs, mask = mask)


# ------------------ Heavy geometric transforms ------------------------

class ElasticTransform:
    """
    Will distort the inputs elastically with
    a probability P=0.5
    """
    def __init__(self, p = 0.5):
        self.transform = A.ElasticTransform(p=p)

    @stack
    def __call__(self, inputs: list, mask: list) -> dict:
        return self.transform(image = inputs, mask = mask)

class GridDistortion:
    """
    Will distort the inputs using grid distortion with
    a probability P=0.5
    """
    def __init__(self, p = 0.5):
        self.transform = A.GridDistortion(p=p)

    @stack
    def __call__(self, inputs: list, mask: list) -> dict:
        return self.transform(image = inputs, mask = mask)

class PincushionDistortion:
    """
    Will distort the inputs pincushion style with
    a probability P=0.5
    """
    def __init__(self, p = 0.5, distort_limit=0.1):
        self.transform = A.OpticalDistortion(p=p, distort_limit=distort_limit)

    @stack
    def __call__(self, inputs: list, mask: list) -> dict:
        return self.transform(image = inputs, mask = mask)

# ----------------Color augmentations--------------------------------

class RandomBrightnessContrast:
    """
    Will adjust the inputs brightness and contrast with
    a probability P=0.5
    """
    def __init__(self, p = 0.5, brightness_limit = 0.01):
        self.transform = A.RandomBrightnessContrast(brightness_limit=brightness_limit, p=p)

    @stack
    def __call__(self, inputs: list, mask: list) -> dict:
        return self.transform(image = inputs, mask = mask)

class ChannelShuffle:
    """
    Will shuffle the input channels with
    a probability P=0.5
    This method assumes channel-first tensors
    """
    def __init__(self, p = 0.5):
        self.p = p


    def __call__(self, input: list, target: list, mask: list) -> tuple:
        if random.random() < self.p:
            indices = [i for i in range(input.shape[0])]
            random.shuffle(indices)
            input = input[indices,:,:]
        return input, target, mask

# ------------Noise injection augmentations-----------------


class GaussianBlur:
    """
    Will inject Gaussian blur into the inputs with
    a probability P=0.5
    """
    def __init__(self, p = 0.5, blur_limit=(3, 7)):
        self.transform = A.GaussianBlur(p=p, blur_limit=blur_limit)

    @stack
    def __call__(self, inputs: list, mask: list) -> dict:
        return self.transform(image = inputs, mask = mask)

class MedianBlur:
    """
    Will inject Median blur into the inputs with
    a probability P=0.5
    """
    def __init__(self, p = 0.5, blur_limit=5):
        self.transform = A.MedianBlur(p=p, blur_limit=blur_limit)

    @stack
    def __call__(self, inputs: list, mask: list) -> dict:
        return self.transform(image = inputs, mask = mask)

# ---------------------- Other --------------------------------------

class ChannelChange:
    """
    A class to be able to change tensor from channels_last to
    channels_first and vice versa
    """
    def __init__(self, changes : tuple):
        """
        changes is a tuple of in which order 
        the channels should be changed
        ex: from channels_first to channels_last: (1,2,0)
        ex: from channels_last to channels_first: (2,0,1)
        """
        self.changes = changes
    
    def __call__(self, input: list, target: list,  mask: list) -> tuple:
        input = np.transpose(input, self.changes)
        target = np.transpose(target, self.changes)
        mask = np.transpose(mask, self.changes)
        return input, target, mask

class RandomResizeCrop:
    """
    A class that crops random parts of the
    inputs to the size crop_size
    """
    def __init__(self, crop_size=(256,256), p=1):
        self.transform = A.RandomResizedCrop(crop_size[1], crop_size[0], scale=(0.08,1.0), ratio=(0.75,1.3333333333333333))

    @stack
    def __call__(self, inputs: list, mask: list) -> dict:
        return self.transform(image = inputs, mask = mask)

# -------------------- Transform compose methods -------------------------

def affine_augmentations():
    """
    The augmentations that we have deemed as
    less risky
    These are custom PyTorch augmentations implemented
    using Albumentations and decorators used for training
    """
    vflip = VerticalFlip()
    hflip = HorizontalFlip()
    rotate = Rotate()
    channels_last = ChannelChange((1,2,0))
    channels_first = ChannelChange((2,0,1))
    transpose = Transpose()
    transforms = A.Compose([
        ChannelChange((1,2,0)),
        RandomResizeCrop(),
        VerticalFlip(),
        HorizontalFlip(),
        Rotate(),
        Transpose(),
        ChannelChange((2,0,1))
        ])
    return transforms

def all_augmentations():
    """
    All augmentations:
    These are experimental custom PyTorch
    augmentations implemented using Albumentations
    """
    vflip = VerticalFlip()
    hflip = HorizontalFlip() 
    rotate = Rotate()
    channels_last = ChannelChange((1,2,0))
    channels_first = ChannelChange((2,0,1))
    transpose = Transpose() 
    elastic = ElasticTransform()
    grid_distortion = GridDistortion()
    brightness_contrast = RandomBrightnessContrast()
    channel_shuffle = ChannelShuffle()
    gaussian_blur = GaussianBlur()
    median_blur = MedianBlur()
    pincushion = PincushionDistortion()
    transforms = A.Compose([
        RandomResizeCrop(),
        ChannelShuffle(),
        ChannelChange((1,2,0)),
        PincushionDistortion(),
        VerticalFlip(),
        HorizontalFlip(),
        Rotate(),
        Transpose() ,
        GaussianBlur(),
        ElasticTransform(),
        GridDistortion(),
        MedianBlur(),
        RandomBrightnessContrast(), # wants the input as float32
        ChannelChange((2,0,1))
        ])
    return transforms

def test_augmentations(crop_size: Tuple[int,int] = (1024,1024)):
    """PyTorch augmentations implemented using Albumentation and decorators
    for the test dataset.
    """
    transforms = A.Compose([
        ChannelChange((1,2,0)),
        RandomResizeCrop(crop_size=crop_size),
        ChannelChange((2,0,1))
        ])
    return transforms
