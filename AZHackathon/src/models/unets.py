from typing import Tuple

import torch
import torch.nn as nn
import torchvision

#from fastai.vision.models.unet import DynamicUnet
import segmentation_models_pytorch as smp

# Documentation for "segmentation_models_pytorch" here
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/unet/model.py

class UnetResnet152(nn.Module):
    def __init__(self, input_channels:int = 7, output_channels:int = 3):
        super(UnetResnet152, self).__init__()

        self.unet = smp.Unet(
            'resnet152', 
            in_channels=input_channels, 
            encoder_weights='imagenet', 
            classes=output_channels,
            encoder_depth=5,
            decoder_channels= (256, 128, 64, 32, 16)
        )

        # Change Up-Sampling kernel size
        for i in range(5):
            self.unet.decoder.blocks[i].conv1[0] = nn.Conv2d(
                self.unet.decoder.blocks[i].conv1[0].in_channels,
                self.unet.decoder.blocks[i].conv1[0].out_channels,
                kernel_size=(5,5),
                stride=self.unet.decoder.blocks[i].conv1[0].stride,
                padding=(2,2),
                bias=self.unet.decoder.blocks[i].conv1[0].bias
            )

            self.unet.decoder.blocks[i].conv2[0] = nn.Conv2d(
                self.unet.decoder.blocks[i].conv2[0].in_channels,
                self.unet.decoder.blocks[i].conv2[0].out_channels,
                kernel_size=(5,5),
                stride=self.unet.decoder.blocks[i].conv2[0].stride,
                padding=(2,2),
                bias=self.unet.decoder.blocks[i].conv2[0].bias
            )

        # Alternative approach: Fastai
        """
        conv_backbone = torchvision.models.resnet152(pretrained=False)

        conv_backbone.conv1 = nn.Conv2d(
            input_channels,
            out_channels = conv_backbone.conv1.out_channels,
            kernel_size = conv_backbone.conv1.kernel_size,
            stride  = conv_backbone.conv1.stride,
            bias  = conv_backbone.conv1.bias
        )
        conv_backbone = nn.Sequential(*list(conv_backbone.children())[:-2])

        # Using Fastai API to construct UNet from any encoders (ResNet152 in this case)
        unet = DynamicUnet( 
            conv_backbone,        # Backbone
            output_channels, # Number of classes/channels
            (256,256),       # Output shape
            norm_type=None
        )
        
        for i in range(1, 12):
            print(i)
            unet.layers[i]
        print(len(unet.layers)) #0-12
        self.unet = unet
        """

    def forward(self, x):
        return self.unet(x)


class UnetResnet152v2(nn.Module):
    def __init__(self, input_channels:int = 7, output_channels:int = 3):
        super(UnetResnet152, self).__init__()

        self.unet = smp.Unet(
            'resnet152', 
            in_channels=input_channels, 
            encoder_weights='imagenet', 
            classes=output_channels,
            encoder_depth=5,
            decoder_channels= (256, 128, 64, 32, 16)
        )

        # Change Up-Sampling kernel size
        print(len(self.unet.decoder.blocks))
        self.unet.decoder.blocks[4].conv2[0] = nn.Conv2d(
            self.unet.decoder.blocks[4].conv2[0].in_channels,
            self.unet.decoder.blocks[4].conv2[0].out_channels,
            kernel_size=(5,5),
            stride=self.unet.decoder.blocks[4].conv2[0].stride,
            padding=(2,2),
            bias=self.unet.decoder.blocks[4].conv2[0].bias
        )
        
    def forward(self, x):
        return self.unet(x)


class UnetDpn92(nn.Module):
    def __init__(self, input_channels:int = 7, output_channels:int = 3):
        super(UnetDpn92, self).__init__()
        self.unet = smp.Unet(
            'dpn92', 
            in_channels=input_channels, 
            encoder_weights='imagenet+5k', 
            classes=output_channels,
            encoder_depth=5,
            decoder_channels= (256, 128, 64, 32, 16)         
        )

        # Change Up-Sampling kernel size
        for i in range(5):
            self.unet.decoder.blocks[i].conv1[0] = nn.Conv2d(
                self.unet.decoder.blocks[i].conv1[0].in_channels,
                self.unet.decoder.blocks[i].conv1[0].out_channels,
                kernel_size=(5,5),
                stride=self.unet.decoder.blocks[i].conv1[0].stride,
                padding=(2,2),
                bias=self.unet.decoder.blocks[i].conv1[0].bias
            )

            self.unet.decoder.blocks[i].conv2[0] = nn.Conv2d(
                self.unet.decoder.blocks[i].conv2[0].in_channels,
                self.unet.decoder.blocks[i].conv2[0].out_channels,
                kernel_size=(5,5),
                stride=self.unet.decoder.blocks[i].conv2[0].stride,
                padding=(2,2),
                bias=self.unet.decoder.blocks[i].conv2[0].bias
            )

    def forward(self, x):
        return self.unet(x)

class UnetResnext101_32x8d(nn.Module):
    def __init__(self, input_channels:int = 7, output_channels:int = 3):
        super(UnetResnext101_32x8d, self).__init__()

        self.unet = smp.Unet(
            'resnext101_32x8d', 
            in_channels=input_channels, 
            encoder_weights='imagenet', 
            classes=output_channels,
            encoder_depth=5,
            decoder_channels= (256, 128, 64, 32, 16)
        )

        # Change Up-Sampling kernel size
        for i in range(5):
            self.unet.decoder.blocks[i].conv1[0] = nn.Conv2d(
                self.unet.decoder.blocks[i].conv1[0].in_channels,
                self.unet.decoder.blocks[i].conv1[0].out_channels,
                kernel_size=(5,5),
                stride=self.unet.decoder.blocks[i].conv1[0].stride,
                padding=(2,2),
                bias=self.unet.decoder.blocks[i].conv1[0].bias
            )

            self.unet.decoder.blocks[i].conv2[0] = nn.Conv2d(
                self.unet.decoder.blocks[i].conv2[0].in_channels,
                self.unet.decoder.blocks[i].conv2[0].out_channels,
                kernel_size=(5,5),
                stride=self.unet.decoder.blocks[i].conv2[0].stride,
                padding=(2,2),
                bias=self.unet.decoder.blocks[i].conv2[0].bias
            )

        # Alternative approach: Fastai
        """ 
        conv_backbone = torchvision.models.resnext101_32x8d(pretrained=False)

        conv_backbone.conv1 = nn.Conv2d(
            input_channels,
            out_channels = conv_backbone.conv1.out_channels,
            kernel_size = conv_backbone.conv1.kernel_size,
            stride  = conv_backbone.conv1.stride,
            bias  = conv_backbone.conv1.bias
        )
        conv_backbone = nn.Sequential(*list(conv_backbone.children())[:-2])

        # Using Fastai API to construct UNet from any encoders (ResNet152 in this case)
        unet = DynamicUnet( 
            conv_backbone,        # Backbone
            output_channels, # Number of classes/channels
            (256,256),       # Output shape
            norm_type=None
        )
        self.unet = unet
        """

    def forward(self, x):
        return self.unet(x)    

if __name__ == "__main__":
    # Testing the models for correcteness
    
    for Model in [UnetResnet152, UnetDpn92, UnetResnext101_32x8d]:
        print(Model)

        model = Model(output_channels=2)
        model.eval()
        with torch.no_grad():

            x = torch.rand(1, 7, 256, 256)
            output = model(x)
            print("Input shape:", x.shape, " --> ", "Output shape:", output.shape)
            print("Output max value:", output.max(), ",", "Output min shape:", output.min())
            
            
            x = torch.rand(1, 7, 512, 512)
            output = model(x)
            print("Input shape:", x.shape, " --> ", "Output shape:", output.shape)
            print("Output max value:", output.max(), ",", "Output min shape:", output.min())

            x = torch.rand(1, 7, 1024, 1024)
            output = model(x)
            print("Input shape:", x.shape, " --> ", "Output shape:", output.shape)
            print("Output max value:", output.max(), ",", "Output min shape:", output.min())
        
        print("\n")
