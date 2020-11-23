import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm

NUM_WORKERS = 4

zero2minusone = {
    'mean': [0.5, 0.5, 0.5],
    'std' : [0.5, 0.5, 0.5]
} 

minusone2zero = {
    'mean': [-1.0, -1.0, -1.0],
    'std' : [2.0, 2.0, 2.0]
}

MASK_PARAMETERS = {
    'mean': [0.4712, 0.4701, 0.4689],
    'std': [0.3324, 0.3320, 0.3319]
    }

MASK_UNNORMALIZE = {
    'mean' : [-mean/std for mean, std in zip(MASK_PARAMETERS['mean'], MASK_PARAMETERS['std'])],
    'std' : [1.0/std for std in MASK_PARAMETERS['std']]
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

def save_image(mode, label_img, mask_img, generate_img, num_iter, save_dir, mask_unnormal, label_unnormal):
    mask_unnormal = transforms.Normalize(mask_unnormal['mean'], mask_unnormal['std'])
    if mode == 'Edit':
        label_unnormal = transforms.Normalize(label_unnormal['mean'], label_unnormal['std'])
        mask = mask_unnormal(make_grid(mask_img, nrow = 1))
        label = label_unnormal(make_grid(label_img, nrow = 1))
        generate = label_unnormal(make_grid(generate_img, nrow = 1))
        result = torch.cat([mask, label, generate], dim=2)
        result = transforms.ToPILImage()(result)
    else:
        mask = mask_unnormal(make_grid(mask_img, nrow = 1))
        label = make_grid(label_img, nrow = 1)
        generate = make_grid(generate_img, nrow = 1)
        result = torch.cat([mask, label, generate], dim=2)
        result = transforms.ToPILImage()(result)
    result.save(os.path.join(save_dir, f'{num_iter}_{mode}.jpg'))

def save_model(mode, num_iter, save_dir, generator, discriminator=None):
    if mode == 'Edit':
        torch.save({
            'generator' : generator.state_dict(),
            'discriminator' : discriminator.state_dict(),
        }, os.path.join(save_dir, f'{num_iter}.pt'))

    else:
        torch.save({
            'generator' : generator.state_dict()
        }, os.path.join(save_dir, f'{num_iter}_map.pt'))
