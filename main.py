import os, argparse, torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from torchsummary import summary
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm, trange
from model import Mapmodule, Editmodule, Discriminator
from module import PerceptualLoss, Reducenoise, G_loss
from dataset import Mask, create_loader
from utils import *
import sys
sys.stdout.flush()

parser = argparse.ArgumentParser()
parser.add_argument('-m', metavar='MODE', dest='mode', default='Edit', choices=['Edit', 'Map', 'Test'], required=True, help='Edit : Train edit module, Map : Train map module, Test : Make image')
parser.add_argument('--checkpoint', metavar='DIR', default=None, 
help='Directory of trained model')
parser.add_argument('--data_dir', metavar='DIR', default='../data/celeba', help='Dataset or Test image directory. In inference, output image will be saved here')
parser.add_argument('--model_dir', metavar='DIR', default='./checkpoint', help='Directory to save your model when training')
parser.add_argument('--result_dir', metavar='DIR', default='./result', help='Directory to save your Input/True/Generate image when training')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--ratio', metavar='Float', type=float, default=0.1, help='Hold-out ratio, default train is 0.1')
parser.add_argument('--batchsize', metavar='Int', type=int, default=64, help='Default is 64')
parser.add_argument('--lr', metavar='Float', type=float, default=0.0002, help='Default is 0.0002')
parser.add_argument('--epoch', metavar='Int', type=int, default=500, help='Default is 500')
args = parser.parse_args()

if args.mode == 'Test':
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
    
    if args.mode in ['Edit', 'Map']:
        trainloader, valloader = create_loader(
            dir=args.data_dir, 
            mode=args.mode,
            mask_transform=mask_transform, img_transform=img_transform, map_transform=map_transform,
            batchsize=args.batchsize, ratio=args.ratio
            ) 
        if args.mode == 'Edit':
            G = Editmodule(in_channels=4).to(device)
        else:
            G = Mapmodule(in_channels=3).to(device)
        G.weight_init(mean=0.0, std=0.02)
        G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(BETA1, BETA2))
    else:
        G_Edit = Editmodule(in_channels=4).to(device)
        G_Edit.load_state_dict(torch.load(args.checkpoint, map_location=device)['generator'])
        G_Map = Mapmodule(in_channels=3).to(device)
        G_Map.load_state_dict(torch.load('/mnt/serverhdd2/jiwook/project/Server/Map.pt', map_location=device)['generator'])

    if args.mode == 'Edit':
        D_whole = Discriminator().to(device)
        D_whole.weight_init(mean=0.0, std=0.02)
        D_whole_optimizer = optim.Adam(D_whole.parameters(), lr=args.lr, betas=(BETA1, BETA2))
        D_mask = Discriminator().to(device)
        D_mask.weight_init(mean=0.0, std=0.02)
        D_mask_optimizer = optim.Adam(D_mask.parameters(), lr=args.lr, betas=(BETA1, BETA2))
        num_iter = 0
        for epoch in range(args.epoch):
            G_loss_list, Dw_loss_list, Dm_loss_list = [], [], []
            pbar = tqdm(trainloader, desc=f'Epoch 0, G: 0.000, D: None')
            for mask, img, map_img in pbar:
                I_input = torch.cat([mask, map_img], dim=1).to(device)
                I_gt = img.to(device)
                I_map = map_img.to(device)
                #Training
                if epoch < 100:
                    G_loss_list.append(G_train(I_input, I_gt, I_map, G, G_optimizer))
                    G_loss = sum(G_loss_list)/(num_iter - epoch*len(trainloader))
                    pbar.set_description(f'Epoch {epoch}, G: {G_loss:.3f}, D: None')
                elif epoch < 300:
                    I_edit = G(I_input)
                    Dw_loss_list.append(D_train(I_edit, I_gt, D_whole, D_whole_optimizer))
                    G_loss_list.append(G_train(I_input, I_gt, I_map, G, G_optimizer, D_whole))
                    Dw_loss = sum(Dw_loss_list)/(num_iter - epoch*len(trainloader))
                    G_loss = sum(G_loss_list)/(num_iter - epoch*len(trainloader))
                    pbar.set_description(f'Epoch {epoch}, G: {G_loss:.3f}, D_whole: {Dw_loss:.3f}')
                else:
                    I_edit = G(I_input)
                    Dw_loss_list.append(D_train(I_edit, I_gt, D_whole, D_whole_optimizer))
                    I_mask = I_gt*(torch.ones_like(I_map) - I_map) + G(I_input)*I_map
                    Dm_loss_list.append(D_train(I_mask, I_gt, D_mask, D_mask_optimizer))
                    G_loss_list.append(G_train(I_input, I_gt, I_map, G, G_optimizer, D_whole, D_mask))
                    Dw_loss = sum(Dw_loss_list)/(num_iter - epoch*len(trainloader))
                    Dm_loss = sum(Dm_loss_list)/(num_iter - epoch*len(trainloader))
                    G_loss = sum(G_loss_list)/(num_iter - epoch*len(trainloader))
                    pbar.set_description(f'Epoch {epoch}, G: {G_loss:.3f}, [D_Whole, D_Mask]: [{Dw_loss:.3f}, {Dm_loss:.3f}]')
                #Save
                num_iter += 1
            if epoch%5 == 0:
                G.eval()
                with torch.no_grad():
                    mask, img, map_img = iter(valloader).next()
                    generate = G(torch.cat([mask, map_img], dim=1).to(device)).cpu()
                save_image(args.mode, img, mask, generate, epoch, args.result_dir, MASK_UNNORMALIZE, minusone2zero)
                save_model(args.mode, epoch, args.model_dir, G, D_whole, D_mask)

    elif args.mode =='Map':
        num_iter = 0
        for epoch in range(args.epoch):
            print(f'Epoch {epoch+1} Start')  
            G.train()
            for mask, map_img in tqdm(trainloader):
                map_img = map_img.to(device)
                mask = mask.to(device)
                #Training G
                G.zero_grad()
                G_result = G(mask)
                bce_loss = BCE_loss(G_result, map_img)
                G_train_loss = bce_loss 
                G_train_loss.backward()
                G_optimizer.step()
                if num_iter % 200 == 0:
                    G.eval()
                    with torch.no_grad():
                        mask, map_img = iter(valloader).next()
                        generate = G(mask.to(device)).cpu()
                        generate = Reducenoise()(generate)
                    save_image(args.mode, map_img, mask, generate, num_iter, args.result_dir, MASK_UNNORMALIZE, minusone2zero)
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
        user_img = mask_transform(user_img).unsqueeze(dim=0).to(device)
        map_img = Reducenoise().to(device)(G_Map(user_img))
        img = torch.cat([user_img, map_img], dim=1)
        result = G_Edit(img).squeeze(dim=0).cpu()
        test_transform(result).save(args.data_dir[:-4]+'_result.jpg')

        

    