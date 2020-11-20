import os, argparse, torch
import torch.optim as optim
import torch.nn as nn
import pytorch_ssim
from PIL import Image
from torchsummary import summary
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import Mapmodule, Generator, Discriminator
from module import PerceptualLoss
from dataset import Mask, create_loader
from utils import save_image, save_model, MASK_PARAMETERS, IMG_UNNORMALIZE 
import sys
sys.stdout.flush()

parser = argparse.ArgumentParser()
parser.add_argument('-m', metavar='MODE', dest='mode', default='train', choices=['train', 'test'], required=True, help='Train : Use hold-out, Test : Make image')
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    G = Generator(in_channels=3).to(device)
    G.weight_init(mean=0.0, std=0.02)
    
    if args.mode == 'train':
        D = Discriminator().to(device)
        D.weight_init(mean=0.0, std=0.02)

        trainloader, valloader = create_loader(
            dir=args.data_dir, 
            mask_transform=mask_transform, img_transform=img_transform, 
            batchsize=args.batchsize, ratio=args.ratio)
        BCE_loss = nn.BCELoss()
        L1_loss = nn.L1Loss()
        Percept_loss = PerceptualLoss()
        SSIM_loss = pytorch_ssim.SSIM(window_size=11)

        G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(BETA1, BETA2))
        D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(BETA1, BETA2))
 
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        D_losses = []
        G_losses = []   
        num_iter = 0
        for epoch in range(args.epoch):
            print(f'Epoch {epoch+1} Start')  
            G.train()
            D.train()
            for mask, img in tqdm(trainloader):
                img = img.to(device)
                mask = mask.to(device)
                #Training D
                D.zero_grad()
                D_real = D(img).squeeze()
                D_real_loss = BCE_loss(D_real, torch.ones(D_real.size()).to(device))
                G_result = G(mask)
                D_fake = D(G_result).squeeze()
                D_fake_loss = BCE_loss(D_fake, torch.zeros(D_fake.size()).to(device))
                D_train_loss = (D_real_loss + D_fake_loss)/2
                D_train_loss.backward()
                D_optimizer.step()
                D_losses.append(D_train_loss)

                #Training G
                G.zero_grad()
                G_result = G(mask)
                D_result = D(G_result).squeeze()
                
                bce_loss = BCE_loss(D_result, torch.ones(D_result.size()).to(device))
                l1_loss = L1_loss(G_result, img)
                ssim_loss = 1 - SSIM_loss(G_result, img)
                percept_loss = Percept_loss(G_result, img)
                G_train_loss = bce_loss + args.l1_lambda*(l1_loss + ssim_loss + percept_loss)
                G_train_loss.backward()

                G_optimizer.step()
                G_losses.append(G_train_loss)
                if num_iter % 200 == 0:
                    G.eval()
                    with torch.no_grad():
                        img, mask = iter(valloader).next()
                        generate = G(mask.to(device)).cpu()
                    save_image(img, mask, generate, num_iter, args.result_dir, IMG_UNNORMALIZE)
                    save_model(G, D, num_iter, args.model_dir)
                num_iter += 1
    else:
        user_img = Image.open(args.data_dir)
        origin_size = user_img.size
        test_transform = transforms.Compose([
            transforms.Normalize(IMG_UNNORMALIZE['mean'], IMG_UNNORMALIZE['std']),
            transforms.ToPILImage(),
            transforms.Resize(origin_size[::-1])
        ])
        G.load_state_dict(torch.load(args.checkpoint)['generator'])
        user_img = transform(user_img).unsqueeze(dim=0).to(device)
        result = G(user_img).squeeze(dim=0).cpu()
        test_transform(result).save(args.data_dir[:-4]+'_result.jpg')

        

    