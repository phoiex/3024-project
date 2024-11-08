import torch
import torchvision.transforms as transforms
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SVHNDataLoader:
    def __init__(self, data_dir, batch_size=16, max_rotation=50, crop_size=32, aspect_ratio_change=0.2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_rotation = max_rotation
        self.crop_size = crop_size
        self.aspect_ratio_change = aspect_ratio_change
        self.train_transforms = A.Compose([
            A.Rotate(limit=self.max_rotation, p=0.5),
            A.RandomResizedCrop(self.crop_size, self.crop_size, scale=(0.8, 1.0), ratio=(1.0 - self.aspect_ratio_change, 1.0 + self.aspect_ratio_change), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Blur(blur_limit=3, p=0.3),  
            A.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
            ToTensorV2(),
        ])
        
        self.test_transforms = A.Compose([
            A.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
            ToTensorV2(),
        ])


    def transform_data(self, image, transform):
        image_np = np.array(image)
        augmented = transform(image=image_np)
        return augmented['image']

    def load_data(self):
       
        train_dataset = SVHN(root=self.data_dir, split='train', download=True, transform=lambda img: self.transform_data(img, self.train_transforms))
        test_dataset = SVHN(root=self.data_dir, split='test', download=True, transform=lambda img: self.transform_data(img, self.test_transforms))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.fc2(x)
        return x



