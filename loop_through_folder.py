import os
from tqdm import tqdm
from mask import create_mask 

folder_path = "/mnt/serverhdd2/jiwook/data/celeba"

num_of_files = 0
num_of_masks = 0
for path in tqdm(os.listdir(folder_path), desc="Number of Iter"):
    subfolder = os.path.join(folder_path, path) 
    if os.path.isfile(subfolder):
        num_of_files += 1
        try:
            create_mask(subfolder)
        except:
            pass
        path_splits = os.path.splitext(subfolder)
        mask = path_splits[0] + '_mask' + path_splits[1]
        if os.path.isfile(mask):
            num_of_masks +=1
    else:
        for f in tqdm(os.listdir(subfolder), desc="Files in subfolder"):
            file = os.path.join(subfolder, f)
            num_of_files += 1
            try:
                create_mask(file)
            except:
                pass
            path_splits = os.path.splitext(file)
            mask = path_splits[0] + '_mask' + path_splits[1]
            if os.path.isfile(mask):
                num_of_masks +=1
            
print("%d files complete, %d files not found face " %(num_of_masks, num_of_files-num_of_masks))