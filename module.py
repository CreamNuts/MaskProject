import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride1=1):
        super(EncoderBlock, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),        
        )
        
    def forward(self, input):
        return self.Encoder(input)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride1=1):
        super(DecoderBlock, self).__init__()
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True), 
        )
    
    def forward(self, input, skip):
        output = self.Decoder(input)
        pad = (skip.size()[3] - output.size()[3], skip.size()[2] - output.size()[2]) #Width, Height
        output = F.pad(output, [pad[0]//2, pad[0] - pad[0]//2, pad[1]//2, pad[1] - pad[1]//2])
        return torch.cat([skip, output], dim=1)

class DiscrimBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2):
        super(DiscrimBlock, self).__init__()
        self.Block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, input):
        return self.Block(input)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.SE = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        batch, channel, _, _ = input.size()
        se_weight = self.AvgPool(input).view(batch, channel)
        se_weight = self.SE(se_weight).view(batch, channel, 1, 1)
        return input * se_weight.expand_as(input)

"""class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.Encoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, input):
        return self.Encoder(input)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.Upsampling = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.Decoder = ConvBlock(in_channels, out_channels)
    
    def forward(self, input, skip):
        output = self.Upsampling(input)
        pad = (skip.size()[3] - output.size()[3], skip.size()[2] - output.size()[2]) #Width, Height
        output = F.pad(output, [pad[0]//2, pad[0] - pad[0]//2, pad[1]//2, pad[1] - pad[1]//2])
        output = torch.cat([skip, output], dim=1)
        return self.Decoder(output)
"""
