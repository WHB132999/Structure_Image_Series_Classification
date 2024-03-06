import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class StructureDataset(Dataset):
    def __init__(self, image_folder, gt_json_file, transform=None, num_classes=7):
        super().__init__()
        self.image_folder = image_folder
        self.transform = transform
        self.num_classes = num_classes

        with open(gt_json_file, 'r') as gt_file:
            self.json_data = json.load(gt_file)

        self.image_filenames = []
        self.labels = []

        for key_i, value_i in self.json_data.items():
            ## The GTs are in order, however images are not, hence reordering images based on the order of labels.
            pure_label_i = int(value_i.split('-')[1].split('_')[0])
            self.labels.append(pure_label_i)
            self.image_filenames.append(key_i)
    
    def __len__(self):
        assert len(self.image_filenames) == len(self.labels), "image nums must be same as label nums!"
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_filenames[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
    


class StructureSeriesDataset(Dataset):
    def __init__(self, image_folder, gt_json_file, transform=None, num_classes=7):
        super().__init__()
        self.image_folder = image_folder
        self.transform = transform
        self.num_classes = num_classes

        with open(gt_json_file, 'r') as gt_file:
            self.json_data = json.load(gt_file)

        self.image_filenames = []
        self.labels = []

        for key_i, value_i in self.json_data.items():
            ## The GTs are in order, however images are not, hence reordering images based on the order of labels.
            pure_label_i = int(value_i.split('-')[1].split('_')[0])
            self.labels.append(pure_label_i)
            self.image_filenames.append(key_i)
    
    def __len__(self):
        assert len(self.image_filenames) == len(self.labels), "image nums must be same as label nums!"
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_filenames[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

