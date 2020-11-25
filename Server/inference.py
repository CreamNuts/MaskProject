import torch
from torchvision import transforms
from model import Mapmodule, Editmodule
from module import Reducenoise

MASK_PARAMETERS = {
    'mean': [0.4712, 0.4701, 0.4689],
    'std': [0.3324, 0.3320, 0.3319]
    }

minusone2zero = {
    'mean': [-1.0, -1.0, -1.0],
    'std' : [2.0, 2.0, 2.0]
}

class MyModel:
    def __init__(self, edit_path, map_path):
        self.Edit = Editmodule(in_channels=4).cuda()
        self.Edit.load_state_dict(torch.load(edit_path)['generator'])
        self.Map = Mapmodule(in_channels=3).cuda()
        self.Map.load_state_dict(torch.load(map_path)['generator'])

    def predict(self, user_img):
        origin_size = user_img.size
        input_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(MASK_PARAMETERS['mean'], MASK_PARAMETERS['std'])
        ])
        output_transform = transforms.Compose([
            transforms.Normalize(minusone2zero['mean'], minusone2zero['std']),
            transforms.ToPILImage(),
            transforms.Resize(origin_size[::-1])
        ])
        with torch.no_grad():
            user_img = input_transform(user_img).unsqueeze(dim=0).cuda()
            map_img = Reducenoise().cuda()(self.Map(user_img))
            img = torch.cat([user_img, map_img], dim=1)
            output = output_transform(self.Edit(img).squeeze(dim=0).cpu())
        return output


