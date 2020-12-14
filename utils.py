import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import pytorch_ssim
from module import G_loss
from parameter import *
from tqdm import tqdm

NUM_WORKERS = 4

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
    else:
        mask = mask_unnormal(make_grid(mask_img, nrow = 1))
        label = make_grid(label_img, nrow = 1)
        generate = make_grid(generate_img, nrow = 1)
        
    result_torch = torch.cat([mask, label, generate], dim=2)
    result = transforms.ToPILImage()(result_torch)
    result.save(os.path.join(save_dir, f'{num_iter}_{mode}.jpg'))
    return result_torch

def save_model(mode, num_iter, save_dir, generator, discriminator_whole=None, discriminator_mask=None):
    if mode == 'Edit':
        torch.save({
            'generator' : generator.state_dict(),
            'discriminator_whole' : discriminator_whole.state_dict(),
            'discriminator_mask' : discriminator_mask.state_dict()
        }, os.path.join(save_dir, f'{num_iter}.pt'))

    else:
        torch.save({
            'generator' : generator.state_dict()
        }, os.path.join(save_dir, f'{num_iter}_map.pt'))

Total_loss = G_loss()
BCE_loss = nn.BCELoss()

def D_train(input_img, fake_img, real_img, D, optimizer):
    D.train()
    D.zero_grad()
    real_img = torch.cat([input_img, real_img], dim=1)
    D_real = D(real_img).squeeze()
    D_real_loss = BCE_loss(D_real, torch.ones(D_real.size()).to(D_real))
    fake_img = torch.cat([input_img, fake_img], dim=1)
    D_fake = D(fake_img).squeeze()
    D_fake_loss = BCE_loss(D_fake, torch.zeros(D_fake.size()).to(D_fake))
    D_train_loss = (D_real_loss + D_fake_loss)/2
    D_train_loss.backward()
    optimizer.step()
    return D_train_loss

def G_train(input_img, real_img, map_img, G, optimizer, D_whole = None, D_mask = None):
    G.train()
    G.zero_grad()
    I_edit = G(input_img)
    I_mask = None
    if D_whole is not None:
        whole_result = D_whole(torch.cat([input_img, I_edit], dim=1)).squeeze()
        bce_whole = BCE_loss(whole_result, torch.ones(whole_result.size()).to(whole_result))
        G_train_loss = 100*Total_loss(I_edit, real_img) + bce_whole
        if D_mask is not None:
            I_mask = real_img*(torch.ones_like(map_img) - map_img) + I_edit*map_img
            mask_result = D_mask(torch.cat([input_img, I_mask], dim=1)).squeeze()
            bce_mask = BCE_loss(mask_result, torch.ones(mask_result.size()).to(mask_result))
            G_train_loss = 100*Total_loss(I_edit, real_img) + 0.3*bce_whole + 0.7*bce_mask
    else:
        G_train_loss = 100*Total_loss(I_edit, real_img)
    G_train_loss.backward()
    optimizer.step()
    return G_train_loss
    
#
