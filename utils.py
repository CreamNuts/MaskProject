import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vgg19_bn
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm

NUM_WORKERS = 4

IMG_PARAMETERS = {
    'mean':[0.4712, 0.4701, 0.4689],
    'std': [0.3324, 0.3320, 0.3319]
    }

IMG_UNNORMALIZE = {
    'mean' : [-mean/std for mean, std in zip(IMG_PARAMETERS['mean'], IMG_PARAMETERS['std'])],
    'std' : [1.0/std for std in IMG_PARAMETERS['std']]
}

def get_parameters(dataset, batchsize):
    '''
    It takes about 5 minutes
    '''
    IMG_PARAMETERS = {
    'mean':[0, 0, 0],
    'std': [0, 0, 0]
    }

    MASK_PARAMETERS = {
    'mean':[0, 0, 0],
    'std': [0, 0, 0]
    }

    dataloader = DataLoader(dataset, batch_size = batchsize, num_workers=NUM_WORKERS)
    count = 0
    for images, masks in tqdm(dataloader):
        for i in range(3):
            img_var = images[:,:,:,i].view(-1)
            mask_var = masks[:,:,:,i].view(-1)
            IMG_PARAMETERS['mean'][i] += img_var.mean()
            MASK_PARAMETERS['mean'][i] += mask_var.mean()
            IMG_PARAMETERS['std'][i] += img_var.std()
            MASK_PARAMETERS['std'][i] += mask_var.std()
        count += 1

    for i in range(3):
        IMG_PARAMETERS['mean'][i] /= count
        IMG_PARAMETERS['std'][i] /= count
        MASK_PARAMETERS['mean'][i] /= count
        MASK_PARAMETERS['std'][i] /= count

    print('Calculation Completed')
    print(f'IMG_PARAMETERS : {IMG_PARAMETERS}')
    print(f'MASK_PARAMETERS : {MASK_PARAMETERS}')
    return IMG_PARAMETERS, MASK_PARAMETERS

def save_image(input_img, mask_img, generate_img, num_iter, save_dir, unnormalize):
    transform = transforms.Compose([
        transforms.Normalize(unnormalize['mean'], unnormalize['std']),
        transforms.ToPILImage()
    ])
    img = make_grid(input_img, nrow = 1)
    mask = make_grid(mask_img, nrow = 1)
    generate = make_grid(generate_img, nrow = 1)
    result = torch.cat([mask, img, generate], dim=2)
    result = transform(result)
    result.save(os.path.join(save_dir, f'{num_iter}.jpg'))

def save_model(generator, discriminator, num_iter, save_dir):
    torch.save({
        'generator' : generator.state_dict(),
        'discriminator' : discriminator.state_dict(),
    }, os.path.join(save_dir, f'{num_iter}.pt'))

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
        