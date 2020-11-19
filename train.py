import os, argparse, torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from torchsummary import summary
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import Generator, Discriminator
from dataset import Mask, create_loader
from utils import save_image, save_model

parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', default='train', choices=['train', 'test'], help='Train : Use Hold-Out, Test : Make Image')
parser.add_argument('--checkpoint', '-c', default=None, help='Directory to load your model')
parser.add_argument('--data_dir', default='../data/celeba', help='Dataset Directory')
parser.add_argument('--model_dir', default='./checkpoint', help='Directory to save your model')
parser.add_argument('--result_dir', default='./result', help='Directory to save your result img')
parser.add_argument('--gpu', default='0')
parser.add_argument('--l1_lambda', default=100)
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--epoch', type=int, default=10)
args = parser.parse_args()

BETA1 = 0.5
BETA2 = 0.999

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use: ',device)

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])
    
    G = Generator().to(device)
    G.weight_init(mean=0.0, std=0.02)
    
    if args.mode == 'train':
        D = Discriminator().to(device)
        D.weight_init(mean=0.0, std=0.02)

        trainloader, valloader = create_loader(dir=args.data_dir, transform=transform, batchsize=args.batchsize)
        BCE_loss = nn.BCELoss()
        L1_loss = nn.L1Loss()

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
            for img, mask in tqdm(trainloader):
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
                G_train_loss = BCE_loss(D_result, torch.ones(D_result.size()).to(device)) + args.l1_lambda*L1_loss(G_result, img)
                G_train_loss.backward()
                G_optimizer.step()
                G_losses.append(G_train_loss)
                if num_iter % 200 == 0:
                    G.eval()
                    with torch.no_grad():
                        img, mask = iter(valloader).next()
                        generate = G(img.to(device)).cpu()
                    save_image(img, mask, generate, num_iter, args.result_dir)
                    save_model(G, D, num_iter, args.model_dir)
                num_iter += 1
    else:
        G.load_state_dict(torch.load(args.checkpoint))#['generator'])
        img = transform(Image.open(args.data_dir)).unsqueeze(dim=0).to(device)
        result = G(img).squeeze(dim=0).cpu()
        transforms.ToPILImage()(result).save(args.data_dir[:-4]+'_result.jpg')

        

    