import albumentations as A
import numpy as np
import random

def stack(augmentation):
    def wrapper(self, input : list, target : list, mask : list) -> tuple:
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
class VerticalFlip:

    def __init__(self, p = 0.5):
        self.transform = A.VerticalFlip(p=p)

    @stack
    def __call__(self, inputs : list, mask : list) -> dict:
        return self.transform(image = inputs, mask = mask)

class HorizontalFlip:

    def __init__(self, p = 0.5):
        self.transform = A.HorizontalFlip(p=p)

    @stack
    def __call__(self, inputs : list, mask : list) -> dict:
        return self.transform(image = inputs, mask = mask)

class Rotate:

    def __init__(self, p = 0.5, limit = 9):
        self.transform = A.Rotate(limit = limit, p=p)

    @stack
    def __call__(self, inputs : list, mask : list) -> dict:
        return self.transform(image = inputs, mask = mask)

class Transpose:

    def __init__(self, p = 0.5):
        self.transform = A.Transpose(p=p)

    @stack
    def __call__(self, inputs : list, mask : list) -> dict:
        return self.transform(image = inputs, mask = mask)

class Scale:

    def __init__(self, size : tuple, p = 0.5):
        h, w = size[0], size[1]
        self.transform = A.RandomResizedCrop(h, w, p=p)

    @stack
    def __call__(self, inputs : list, mask : list) -> dict:
        return self.transform(image = inputs, mask = mask)

# ------------------ Heavy geometric transforms ------------------------

class ElasticTransform:

    def __init__(self, p = 0.5):
        self.transform = A.ElasticTransform(p=p)

    @stack
    def __call__(self, inputs : list, mask : list) -> dict:
        return self.transform(image = inputs, mask = mask)

class GridDistortion:

    def __init__(self, p = 0.5):
        self.transform = A.GridDistortion(p=p)

    @stack
    def __call__(self, inputs : list, mask : list) -> dict:
        return self.transform(image = inputs, mask = mask)

class PincushionDistortion:

    def __init__(self, p = 0.5, distort_limit=0.1):
        self.transform = A.OpticalDistortion(p=p, distort_limit=distort_limit)

    @stack
    def __call__(self, inputs : list, mask : list) -> dict:
        return self.transform(image = inputs, mask = mask)

# ----------------Color augmentations--------------------------------
class RandomBrightnessContrast:

    def __init__(self, p = 0.5, brightness_limit = 0.01):
        self.transform = A.RandomBrightnessContrast(brightness_limit=brightness_limit, p=p)

    @stack
    def __call__(self, inputs : list, mask : list) -> dict:
        return self.transform(image = inputs, mask = mask)

class HueSaturation:
    pass
    # Not implemented yet due to channel errors in Albumentations

class ChannelShuffle:

    def __init__(self, p = 0.5):
        self.p = p

    # Method assumes channel-first tensors
    def __call__(self, input : list, target : list, mask : list) -> tuple:
        if random.random() < self.p:
            indices = [i for i in range(input.shape[0])]
            random.shuffle(indices)
            input = input[indices,:,:]
        return input, target, mask

# ------------Noise injection augmentations-----------------
class GaussianBlur:

    def __init__(self, p = 0.5, blur_limit=(3, 7)):
        self.transform = A.GaussianBlur(p=p, blur_limit=blur_limit)

    @stack
    def __call__(self, inputs : list, mask : list) -> dict:
        return self.transform(image = inputs, mask = mask)

class MedianBlur:

    def __init__(self, p = 0.5, blur_limit=5):
        self.transform = A.MedianBlur(p=p, blur_limit=blur_limit)

    @stack
    def __call__(self, inputs : list, mask : list) -> dict:
        return self.transform(image = inputs, mask = mask)

# ---------------------- Other --------------------------------------

class ChannelChange:

    def __init__(self, changes):
        self.changes = changes
    
    def __call__(self, input : list, target : list,  mask : list) -> tuple:
        input = np.transpose(input, self.changes)
        target = np.transpose(target, self.changes)
        mask = np.transpose(mask, self.changes)
        return input, target, mask
    
class RandomResizeCrop:

    def __init__(self, crop_size=(256,256), p=1):
        self.transform = A.RandomResizedCrop(crop_size[1], crop_size[0], scale=(0.08,1.0), ratio=(0.75,1.3333333333333333))

    @stack
    def __call__(self, inputs : list, mask : list) -> dict:
        return self.transform(image = inputs, mask = mask)

# -------------------- Transform compose methods -------------------------

def affine_augmentations():
    vflip = VerticalFlip()
    hflip = HorizontalFlip() 
    rotate = Rotate()
    channels_last = ChannelChange((1,2,0))
    channels_first = ChannelChange((2,0,1))
    transpose = Transpose() 
    transforms = A.Compose([
        channels_last,
        RandomResizeCrop(),
        vflip,
        hflip,
        rotate,
        transpose,
        channels_first
        ])
    return transforms

def all_augmentations():
    vflip = VerticalFlip()
    hflip = HorizontalFlip() 
    rotate = Rotate()
    channels_last = ChannelChange((1,2,0))
    channels_first = ChannelChange((2,0,1))
    transpose = Transpose() 
    scale = Scale((512,512))
    elastic = ElasticTransform()
    grid_distortion = GridDistortion()
    brightness_contrast = RandomBrightnessContrast() # wants the input as float32
    channel_shuffle = ChannelShuffle()
    gaussian_blur = GaussianBlur()
    median_blur = MedianBlur()
    pincushion = PincushionDistortion()
    transforms = A.Compose([
        RandomResizeCrop(),
        channel_shuffle,
        channels_last,
        pincushion,
        vflip,
        hflip,
        rotate,
        transpose,
        gaussian_blur,
        elastic,
        grid_distortion,
        median_blur,
        brightness_contrast,
        channels_first
        ])
    return transforms

def test_augmentations(crop_size=(1024,1024)):
    channels_last = ChannelChange((1,2,0))
    channels_first = ChannelChange((2,0,1))
    transforms = A.Compose([
        channels_last,
        RandomResizeCrop(crop_size=crop_size),
        channels_first
        ])
    return transforms
