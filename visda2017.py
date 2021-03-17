import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VisDA17(Dataset):

    def __init__(self, txt_file, root_dir, transform=transforms.ToTensor(), label_one_hot=False, portion=1.0):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.lines = open(txt_file, 'r').readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.label_one_hot = label_one_hot
        self.portion = portion
        self.number_classes = 12
        assert portion != 0
        if self.portion > 0:
            self.lines = self.lines[:round(self.portion * len(self.lines))]
        else:
            self.lines = self.lines[round(self.portion * len(self.lines)):]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = str.split(self.lines[idx])
        path_img = os.path.join(self.root_dir, line[0])
        image = Image.open(path_img)
        image = image.convert('RGB')
        if self.label_one_hot:
            label = np.zeros(12, np.float32)
            label[np.asarray(line[1], dtype=np.int)] = 1
        else:
            label = np.asarray(line[1], dtype=np.int)
        label = torch.from_numpy(label)
        if self.transform:
            image = self.transform(image)
        return image, label