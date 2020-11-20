import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

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

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19_bn(pretrained=True)
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        for block in blocks:
            for layer in block:
                layer.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = transforms.Compose([
            transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
        ])
        
    def forward(self, input, target):
        if input.get_device() != next(self.blocks[0].parameters()).get_device():
            self.blocks = self.blocks.to(input.get_device())
        input = F.interpolate(input, size=224)
        target = F.interpolate(target, size=224)
        loss = 0
        for block in self.blocks:
            input = block(input)
            target = block(target)
            loss += F.l1_loss(input, target)
        return loss
        
class Morphology(nn.Module):
    '''
    Base class for morpholigical operators 
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure. 
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)
        
        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))

        # erosion
        weight = self.weight.view(self.out_channels, -1) # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)

        if self.type == 'erosion2d':
            x = weight - x # (B, Cout, Cin*kH*kW, L)
        elif self.type == 'dilation2d':
            x = weight + x # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError
        
        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False) # (B, Cout, L)
        else:
            x = torch.logsumexp(x*self.beta, dim=2, keepdim=False) / self.beta # (B, Cout, L)

        if self.type == 'erosion2d':
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return x 

class Dilation2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'dilation2d')

class Erosion2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'erosion2d')



def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

    
# class EncoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(EncoderBlock, self).__init__()
#         self.Encoder = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             ConvBlock(in_channels, out_channels)
#         )

#     def forward(self, input):
#         return self.Encoder(input)

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DecoderBlock, self).__init__()
#         self.Upsampling = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
#         self.Decoder = ConvBlock(in_channels, out_channels)
    
#     def forward(self, input, skip):
#         output = self.Upsampling(input)
#         pad = (skip.size()[3] - output.size()[3], skip.size()[2] - output.size()[2]) #Width, Height
#         output = F.pad(output, [pad[0]//2, pad[0] - pad[0]//2, pad[1]//2, pad[1] - pad[1]//2])
#         output = torch.cat([skip, output], dim=1)
#         return self.Decoder(output)
