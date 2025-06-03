# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os

from torchvision import transforms

class FewshotDataset(data.Dataset):
    """
    Args:
        csv_path: csv file path
        transform: data augmentation

    """
    def __init__(self, csv_path, transform=None):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            img_path, name, wnid = l.split(',')
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(img_path)
            label.append(lb)

        self.x = data
        self.y = label

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((84, 84)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        path, label = self.x[i], self.y[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


class OceanDataset(data.Dataset):
    """
    Args:
        csv_path: csv file path
        transform: data augmentation
        sub_transform: data augmentation for subimage

    多兩個輸出, subimage

    """
    def __init__(self, csv_path, transform, sub_transform):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            img_path, name, wnid = l.split(',')
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(img_path)
            label.append(lb)

        self.x = data
        self.y = label

        self.transform = transform
        self.sub_transform = sub_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        path, label = self.x[i], self.y[i]
        img = Image.open(path).convert('RGB')
        image = self.transform(img)

        sub_1 = self.sub_transform( img)
        sub_2 = self.sub_transform( img)

        return image, sub_1, sub_2, label