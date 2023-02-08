import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

class DSB2018Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, train=True, transform=None):
        self.image_dir = image_dir
        self.img_id = os.listdir(image_dir)
        self.mask_dir = mask_dir
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        if self.train:
            img_dir = os.path.join(self.image_dir, self.img_id[idx], 'image.png')
            mask_dir = os.path.join(self.mask_dir, self.img_id[idx], 'mask.png')
            img = Image.open(img_dir).convert('RGB')
            mask = Image.open(mask_dir, as_gray=True).astype(np.bool)
        else:
            img_dir = os.path.join(self.root_dir, self.img_id[idx], 'image.png')
            img = Image.open(img_dir).convert('RGB')
            return {'image':img}
            # size = (img.shape[0],img.shape[1])  # (Height, Weidth)

        if self.transform:
            image, target = self.transform(image, target)

        return {'image':img, 'mask':target}