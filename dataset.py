import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class Mask(Dataset):
    def __init__(self, dir='../data/celeba', mode='train', img_transform=None, mask_transform=None):
        #print(f'creating data loader - {mode}')
        #assert mode in ['train', 'val', 'test']
        #self.mode = mode
        self.image_paths = []
        self.mask_paths = sorted(list(Path(dir).rglob('*_mask.jpg')))
        for image_path in self.mask_paths:
            fullpath = Path(dir)/image_path.name.replace("_mask", "") 
            self.image_paths.append(fullpath)
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        print(f'Total Number of Image is {len(self.image_paths)}')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.img_transform is not None:
            image = self.img_transform(image)
        
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return image, mask