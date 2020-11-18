import os, argparse, model, torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from dataset import Mask
from utils import *
from tqdm import tqdm

IMG_PARAMETERS = {
    'mean':[0.4712, 0.4701, 0.4689],
    'std': [0.3324, 0.3320, 0.3319]
    }

MASK_PARAMETERS = {
    'mean':[0.4712, 0.4701, 0.4689],
    'std': [0.3324, 0.3320, 0.3319]
    }

BATCHSIZE = 128
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
TRAINING_EPOCH = 10
L1_LAMBDA = 100

transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_PARAMETERS['mean'], IMG_PARAMETERS['std']),
])

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use: ',device)

dir = '../data/celeba'
dataset = Mask(dir, img_transform=transform, mask_transform=transform)
num_train = int(len(dataset) * 0.8)
num_valid = len(dataset) - num_train
trainset, valset = random_split(dataset, [num_train, num_valid])

trainloader = DataLoader(trainset, shuffle=True, batch_size=BATCHSIZE)
valloader = DataLoader(valset, batch_size=BATCHSIZE)

G = model.generator()
D = model.discriminator()
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()
G.train()
D.train()

BCE_loss = nn.BCELoss()
L1_loss = nn.L1Loss()

G_optimizer = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
D_optimizer = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []

num_iter = 0
D_losses = []
G_losses = []
for epoch in range(TRAINING_EPOCH):  
    for img, mask in tqdm(trainloader):
        print(img.shape)
        break
        img = img.to(device)
        mask = mask.to(device)
        #Training D
        D.zero_grad()
        D_result = D(img, mask).squeeze()
        D_real_loss = BCE_loss(D_result, torch.ones(D_result.size()).to(device))

        G_result = G(img)
        D_result = D(img, G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, torch.zeros(D_result.size()).to(device))

        D_train_loss = (D_real_loss + D_fake_loss)/2
        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data[0])

        #Training G
        G.zero_grad()

        G_result = G(img)
        D_result = D(img, G_result).squeeze()

        G_train_loss = BCE_loss(D_result, torch.ones(D_result.size().to(device))) + L1_LAMBDA*L1_loss(G_result, mask)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])

        num_iter += 1
    torch.save(G.state_dict(), 'G.pt')
    torch.save(D.state_dict(), 'D.pt')