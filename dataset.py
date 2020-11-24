import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, DataLoader


class Mask(Dataset):
    def __init__(self, dir, mode, mask_transform=None, img_transform=None, map_transform=None):
        self.image_paths = []
        self.mask_paths = sorted(list(Path(dir).rglob('*_mask.jpg')))
        self.map_paths = sorted(list(Path(dir).rglob('*_maskgt.jpg')))
        assert len(self.mask_paths) == len(self.map_paths), 'Some Images doesn\'t have Map Image'
        for image_path in self.mask_paths:
            fullpath = Path(dir)/image_path.name.replace("_mask", "") 
            self.image_paths.append(fullpath)
        self.mask_transform = mask_transform
        self.img_transform = img_transform
        self.map_transform = map_transform
        self.mode = mode
        print(f'In {self.mode} mode, Total Number of Images is {len(self.image_paths)}')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        map_path = self.map_paths[idx]
        image_path = self.image_paths[idx]
        mask = Image.open(mask_path)
        map = Image.open(map_path)  
        if self.map_transform is not None:
            map = self.map_transform(map)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        if self.mode == 'Edit':
            image = Image.open(image_path) 
            if self.img_transform is not None:
                image = self.img_transform(image)
                return mask, image, map
        else:
            return mask, map

def create_loader(dir, mode, mask_transform, img_transform, map_transform, batchsize, ratio=0.8):
    dataset = Mask(dir=dir, mode=mode, mask_transform=mask_transform, img_transform=img_transform, map_transform=map_transform)
    num_train = int(len(dataset) * ratio)
    num_valid = len(dataset) - num_train
    trainset, valset = random_split(dataset, [num_train, num_valid])
    print(f'Number of [Train Images, Valid Images] is [{len(trainset)}, {len(valset)}]')
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batchsize)
    valloader = DataLoader(valset, batch_size=4)
    return trainloader, valloader