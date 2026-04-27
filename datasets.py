import torch
from torch.utils.data import Dataset
import os
import numpy as np

class DSpritesDataset(Dataset):
    def __init__(self, img_path, device=torch.device('cpu')):
        self.device = device
        self.load_data(img_path)

    def load_data(self, img_path):
        print("Loading DSprites Dataset...")
        dsprites_data_path = os.path.join(
            os.getcwd(), img_path)
        dsprites_data = np.load(dsprites_data_path)
        self.data = torch.from_numpy(
            dsprites_data['imgs']).float().unsqueeze(1).to(self.device)
        print("Loaded DSprites Dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
