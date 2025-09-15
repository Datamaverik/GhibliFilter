from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class RealGhibliDataset(Dataset):
    def __init__(self, root_ghibli, root_real, transform=None):
        self.root_ghibli = root_ghibli
        self.root_real = root_real
        self.transform = transform
        
        self.ghilbi_images = os.listdir(root_ghibli)
        self.real_images = os.listdir(root_real)
        self.length_dataset = max(len(self.ghilbi_images), len(self.real_images))
        self.ghilbi_len = len(self.ghilbi_images)
        self.real_len = len(self.real_images)
    
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        ghilbi_img_name = self.ghilbi_images[index % self.ghilbi_len]
        real_img_name = self.real_images[index % self.real_len]

        ghilbi_path = os.path.join(self.root_ghibli, ghilbi_img_name)
        real_path = os.path.join(self.root_real, real_img_name)

        ghilbi_img = Image.open(ghilbi_path).convert("RGB")
        real_img = Image.open(real_path).convert("RGB")

        # Ensure both images are the same size before transform
        target_size = (256, 256)  # Should match config.py transforms
        ghilbi_img = ghilbi_img.resize(target_size, Image.BICUBIC)
        real_img = real_img.resize(target_size, Image.BICUBIC)

        ghilbi_img = np.array(ghilbi_img)
        real_img = np.array(real_img)

        if self.transform:
            augumentations = self.transform(image=ghilbi_img, image0=real_img)
            ghilbi_img = augumentations["image"]
            real_img = augumentations["image0"]

        return ghilbi_img, real_img
        
        