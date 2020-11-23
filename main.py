import os, argparse, torch
import torch.optim as optim
import torch.nn as nn
import pytorch_ssim
from PIL import Image
from torchsummary import summary
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm, trange
from model import Mapmodule, Editmodule, Discriminator
from module import PerceptualLoss, Reducenoise
from dataset import Mask, create_loader
from utils import *
import sys
sys.stdout.flush()

parser = argparse.ArgumentParser()
parser.add_argument('-m', metavar='MODE', dest='mode', default='Edit', choices=['Edit', 'Map', 'test'], required=True, help='Edit : Train edit module, Map : Train map module, Test : Make image')
parser.add_argument('--checkpoint', metavar='DIR', default=None, 
help='Directory of trained model')
parser.add_argument('--data_dir', metavar='DIR', default='../data/celeba', help='Dataset or Test image directory. In inference, output image will be saved here')
parser.add_argument('--model_dir', metavar='DIR', default='./checkpoint', help='Directory to save your model when training')
parser.add_argument('--result_dir', metavar='DIR', default='./result', help='Directory to save your Input/True/Generate image when training')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--l1_lambda', metavar='Float', type=float, default=100, help='Default is 100')
parser.add_argument('--ratio', metavar='Float', type=float, default=0.8, help='Hold-out ratio, default is 0.8')
parser.add_argument('--batchsize', metavar='Int', type=int, default=64, help='Default is 64')
parser.add_argument('--lr', metavar='Float', type=float, default=0.0002, help='Default is 0.0002')
parser.add_argument('--epoch', metavar='Int', type=int, default=10, help='Default is 10')
args = parser.parse_args()

if args.mode == 'test':
    assert os.path.isfile(args.data_dir), 'In testing, data_dir is a file, not a directory'
BETA1 = 0.5
BETA2 = 0.999

if args.gpu == None:
    device = torch.device('cpu')
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use: ',device)

if __name__ == '__main__':
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(MASK_PARAMETERS['mean'], MASK_PARAMETERS['std'])
        ])
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(zero2minusone['mean'], zero2minusone['std'])
    ])
    map_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    if args.mode in ['Edit', 'test']:
        G = Editmodule(in_channels=4).to(device)
    else:
        G = Mapmodule(in_channels=3).to(device)
    G.weight_init(mean=0.0, std=0.02)
    if args.mode != 'test':
        trainloader, valloader = create_loader(
            dir=args.data_dir, 
            mode=args.mode,
            mask_transform=mask_transform, img_transform=img_transform, map_transform=map_transform,
            batchsize=args.batchsize, ratio=args.ratio
            ) 
    BCE_loss = nn.BCELoss()
    L1_loss = nn.L1Loss()
    Percept_loss = PerceptualLoss()
    SSIM_loss = pytorch_ssim.SSIM(window_size=11)
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(BETA1, BETA2))
    G_losses = []   
    if args.mode == 'Edit':
        D_losses = []   
        D = Discriminator().to(device)
        D.weight_init(mean=0.0, std=0.02)
        D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(BETA1, BETA2))
        num_iter = 0
        pbar = tqdm(trainloader, desc='Epoch 0, BL: 0.000, L1: 0.000, SS: 0.000, PL: 0.000')
        for epoch in range(args.epoch):
            G.train()
            D.train()
            for mask, img, map in pbar:
                img = img.to(device)
                input_img = torch.cat([mask, map], dim=1).to(device)
                #Training D
                D.zero_grad()
                D_real = D(img).squeeze()
                D_real_loss = BCE_loss(D_real, torch.ones(D_real.size()).to(device))
                G_result = G(input_img)
                D_fake = D(G_result).squeeze()
                D_fake_loss = BCE_loss(D_fake, torch.zeros(D_fake.size()).to(device))
                D_train_loss = (D_real_loss + D_fake_loss)/2
                D_train_loss.backward()
                D_optimizer.step()
                D_losses.append(D_train_loss)

                #Training G
                G.zero_grad()
                G_result = G(input_img)
                D_result = D(G_result).squeeze()
                bce_loss = BCE_loss(D_result, torch.ones(D_result.size()).to(device))
                l1_loss = L1_loss(G_result, img)
                ssim_loss = 1 - SSIM_loss(G_result, img)
                percept_loss = Percept_loss(G_result, img)
                G_train_loss = bce_loss + args.l1_lambda*(l1_loss + ssim_loss + percept_loss)
                G_train_loss.backward()
                pbar.set_description(f'Epoch {epoch}, BL: {bce_loss:.3f}, L1: {l1_loss:.3f}, SS: {ssim_loss:.3f}, PL: {percept_loss:.3f}')
                G_optimizer.step()
                G_losses.append(G_train_loss)
                if num_iter % 200 == 0:
                    G.eval()
                    with torch.no_grad():
                        mask, img, map = iter(valloader).next()
                        generate = G(torch.cat([mask, map], dim=1).to(device)).cpu()
                    save_image(args.mode, img, mask, generate, num_iter, args.result_dir, MASK_UNNORMALIZE, minusone2zero)
                    save_model(args.mode, num_iter, args.model_dir, G, D)
                num_iter += 1

    elif args.mode =='Map':
        for epoch in range(args.epoch):
            print(f'Epoch {epoch+1} Start')  
            G.train()
            num_iter = 0
            for mask, map in tqdm(trainloader):
                map = map.to(device)
                mask = mask.to(device)
                #Training G
                G.zero_grad()
                G_result = G(mask)
                
                bce_loss = BCE_loss(G_result, map)
                #l1_loss = L1_loss(G_result, map)
                #ssim_loss = 1 - SSIM_loss(G_result, map)
                G_train_loss = bce_loss 
                # + args.l1_lambda*(l1_loss + ssim_loss)
                G_train_loss.backward()

                G_optimizer.step()
                G_losses.append(G_train_loss)
                if num_iter % 200 == 0:
                    G.eval()
                    with torch.no_grad():
                        mask, map = iter(valloader).next()
                        generate = G(mask.to(device)).cpu()
                        generate = Reducenoise()(generate)
                    save_image(args.mode, map, mask, generate, num_iter, args.result_dir, MASK_UNNORMALIZE, minusone2zero)
                    save_model(args.mode, num_iter, args.model_dir, G)
                num_iter += 1
        
    else:
        user_img = Image.open(args.data_dir).convert('RGB')
        origin_size = user_img.size
        test_transform = transforms.Compose([
            transforms.Normalize(minusone2zero['mean'], minusone2zero['std']),
            transforms.ToPILImage(),
            transforms.Resize(origin_size[::-1])
        ])
        G.load_state_dict(torch.load(args.checkpoint)['generator'])
        user_img = mask_transform(user_img).unsqueeze(dim=0).to(device)
        result = G(user_img).squeeze(dim=0).cpu()
        test_transform(result).save(args.data_dir[:-4]+'_result.jpg')

        

    