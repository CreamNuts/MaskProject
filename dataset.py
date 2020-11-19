import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, DataLoader


class Mask(Dataset):
    def __init__(self, dir, transform=None):
        self.image_paths = []
        self.mask_paths = sorted(list(Path(dir).rglob('*_mask.jpg')))
        for image_path in self.mask_paths:
            fullpath = Path(dir)/image_path.name.replace("_mask", "") 
            self.image_paths.append(fullpath)
        self.transform = transform
        print(f'Total Number of Images is {len(self.image_paths)}')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

def create_loader(dir, transform, batchsize, ratio=0.8):
    dataset = Mask(dir=dir, transform=transform)
    num_train = int(len(dataset) * ratio)
    num_valid = len(dataset) - num_train
    trainset, valset = random_split(dataset, [num_train, num_valid])
    print(f'Number of [Train Images, Valid Images] is [{len(trainset)}. {len(valset)}]')
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batchsize)
    valloader = DataLoader(valset, batch_size=4)
    return trainloader, valloader