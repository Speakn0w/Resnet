from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import json
import albumentations as A

class ClassificationDataset(Dataset):
    def __init__(self, root_dir, json_path, transform = None):
        super(ClassificationDataset,self).__init__()
        self.root_dir = root_dir
        self.json_path = json_path
        self.transform = transform
        self.classes_for_all_imgs = []   ##类别标签，用于重采样

        with open(self.json_path, 'r') as file:
            self.json_dict = json.load(open(self.json_path,'r'))
            for item in self.json_dict:
                item['path'] = item['path'].replace('\\', '/')
                self.classes_for_all_imgs.append(int(item['label']))

    def __len__(self):
        return 2*len(self.json_dict)

    def __getitem__(self, index):
        label = self.json_dict[index % len(self.json_dict)]['label']
        img = self.json_dict[index % len(self.json_dict)]['path']
        img_path = os.path.join(self.root_dir, img)
        img = np.array(Image.open(img_path).convert('RGB'))

        size = min(np.size(img,0),np.size(img,1))
        A.center_crop(img, size, size)
        if self.transform:
            augmentations = self.transform(image=img)
            img = augmentations["image"]

        return img, label

    def get_classes_for_all_imgs(self):
        return self.classes_for_all_imgs