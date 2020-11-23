from module import *

class Mapmodule(nn.Module):
    # input shape : (1, 256, 256)
    # initializers
    def __init__(self, in_channels=3, out_channels=1, d=64):
        super(Mapmodule, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1),
        )
        self.encoder2 = EncoderBlock(d, d*2, stride1=2)
        self.encoder3 = EncoderBlock(d*2, d*4, stride1=2)
        self.encoder4 = EncoderBlock(d*4, d*8, stride1=2)
        self.encoder5 = EncoderBlock(d*8, d*16, stride1=2)
        self.decoder1 = DecoderBlock(d*16, d*8, stride1=2)
        self.decoder2 = DecoderBlock(d*8*2, d*4, stride1=2)
        self.decoder3 = DecoderBlock(d*4*2, d*2, stride1=2)
        self.decoder4 = DecoderBlock(d*2*2, d, stride1=2)
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(d*2, d, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(d, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        e1 = self.encoder1(input)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        output = self.decoder1(e5, e4)
        output = self.decoder2(output, e3)
        output = self.decoder3(output, e2)
        output = self.decoder4(output, e1)
        output = self.decoder5(output)
        return output
        
class Editmodule(nn.Module):
    # input shape : (3, 256, 256)
    # initializers
    def __init__(self, in_channels=4, out_channels=3, d=64):
        super(Editmodule, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1),
        )
        self.encoder2 = EncoderBlock(d, d*2, stride1=2)
        self.encoder3 = nn.Sequential(
            EncoderBlock(d*2, d*4, stride1=2),
            SEBlock(d*4, d*4)
        )
        self.encoder4 = nn.Sequential(
            EncoderBlock(d*4, d*8, stride1=2),
            SEBlock(d*8, d*8)
        )
        self.encoder5 = nn.Sequential(
            EncoderBlock(d*8, d*16, stride1=2),
            SEBlock(d*16, d*16)
        )

        self.atrous = nn.Sequential(
            nn.Conv2d(d*16, d*16, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.Conv2d(d*16, d*16, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.Conv2d(d*16, d*16, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.Conv2d(d*16, d*16, kernel_size=3, stride=1, padding=16, dilation=16),
        )

        self.decoder1 = DecoderBlock(d*16, d*8, stride1=2)
        self.decoder2 = DecoderBlock(d*8*2, d*4, stride1=2)
        self.decoder3 = DecoderBlock(d*4*2, d*2, stride1=2)
        self.decoder4 = DecoderBlock(d*2*2, d, stride1=2)
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(d*2, d, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(d, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        e1 = self.encoder1(input)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e5 = self.atrous(e5)
        output = self.decoder1(e5, e4)
        output = self.decoder2(output, e3)
        output = self.decoder3(output, e2)
        output = self.decoder4(output, e1)
        output = self.decoder5(output)
        return output

class Discriminator(nn.Module):
    # initializers
    def __init__(self, in_channels=3, d=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            DiscrimBlock(d, d*2),
            DiscrimBlock(d*2, d*4),
            DiscrimBlock(d*4, d*8),
            nn.Conv2d(d*8, 1, kernel_size=4, padding=1),
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        output = self.model(input)
        output = F.avg_pool2d(output, output.size()[2:]).view(output.size()[0], -1)
        return torch.sigmoid(output)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()