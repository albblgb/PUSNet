import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import torch
import os
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import cv2
from PIL import Image
# import sys 
# sys.path.append(".")
import config as c

class dataset_(Dataset):
    def __init__(self, img_dir, transform, sigma):
        self.img_dir = img_dir
        self.img_filenames = list(sorted(os.listdir(img_dir)))
        self.transform = transform
        self.sigma = sigma
        self.totensor = T.ToTensor()
    
    def __len__(self):
        return len(self.img_filenames)
    
    def __getitem__(self, index):
        img_paths = os.path.join(self.img_dir, self.img_filenames[index])
        img = Image.open(img_paths).convert("RGB")
        img = self.transform(img)
        if self.sigma != None:
            noised_img = img + torch.randn(img.shape).mul_(self.sigma/255)
            return img, noised_img
        return img
    

transform_train = T.Compose([
    T.RandomCrop(c.crop_size_train),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    # T.RandomCrop(c.crop_size_train),
    T.ToTensor()
])

transform_val = T.Compose([
    # T.CenterCrop(c.crop_size_train),
    T.Resize([c.resize_size_test, c.resize_size_test]),
    T.ToTensor(),
])


def load_dataset(train_data_dir, test_data_dir, batchsize_train, batchsize_test, sigma=None):

    train_loader = DataLoader(
        dataset_(train_data_dir, transform_train, sigma),
        batch_size=batchsize_train,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset_(test_data_dir, transform_val, sigma),
        batch_size=batchsize_test,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        drop_last=True
    )

    return train_loader, test_loader


# transform_train = A.Compose(
#     [
#         A.RandomCrop(128, 128),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         ToTensorV2(),
#     ]
# )

# transform_val = A.Compose([
#     A.CenterCrop(256, 256),
#     ToTensorV2(),
# ])


# class dataset_(Dataset):
#     def __init__(self, img_dir, sigma, transform):
#         self.img_dir = img_dir
#         self.img_filenames = list(sorted(os.listdir(img_dir)))
#         self.sigma = sigma
#         self.transform = transform

#     def __len__(self):
#         return len(self.img_filenames)

#     def __getitem__(self, idx):
#         img_filename = self.img_filenames[idx]
#         img = cv2.imread(os.path.join(self.img_dir, img_filename))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.float32(img/255)
        
#         if self.transform:
#             img = self.transform(image=img)["image"]
            
#         noised_img = img + torch.randn(img.shape).mul_(self.sigma/255)

#         return img, noised_img


# def load_dataset(train_data_dir, test_data_dir, batch_size, sigma=None):

#     train_loader = DataLoader(
#         dataset_(train_data_dir, sigma, transform_train),
#         batch_size=batch_size,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=8,
#         drop_last=True
#     )

#     test_loader = DataLoader(
#         dataset_(test_data_dir, sigma, transform_val),
#         batch_size=2,
#         shuffle=False,
#         pin_memory=True,
#         num_workers=1,
#         drop_last=True
#     )

#     return train_loader, test_loader


    
