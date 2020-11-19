import multiprocessing
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from dataset import *

NUM_WORKERS = 4 #multiprocessing.cpu_count()

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

def make_result(input_img, mask_img, generate_img, num_iter, save_dir='./result'):
    img = make_grid(input_img, nrow = 1)
    mask = make_grid(mask_img, nrow = 1)
    generate = make_grid(generate_img, nrow = 1)
    result = torch.cat([img, mask, generate], dim=2)
    result = transforms.ToPILImage()(result)
    result.save(save_dir+f'/{num_iter}_iter.jpg')

    