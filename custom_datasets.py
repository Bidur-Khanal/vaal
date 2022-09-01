import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
from collections import namedtuple
import numpy
import PIL.Image as Image
from utils import *
import math
from exr_data import readEXR
import random

class LiverSegDataset(Dataset):


    def __init__(self, root_dir):

        # Based on https://github.com/mcordts/cityscapesScripts
        Seg_classes = namedtuple('Class', ['name', 'train_id','color', 'color_name'])
        self.classes = [
        Seg_classes('liver', 0, (0, 0, 255), 'blue'),
        Seg_classes('stomach', 1, (255, 0, 0), 'red'),
        Seg_classes('abdominal wall', 2, (0, 255, 0), 'green'),
        Seg_classes('gallbladder', 3, (0, 255, 255), 'cyan'),
        Seg_classes('ligament', 4, (255, 0, 255), 'magenta')]

        self.colors = [c.color for c in self.classes ]
        self.train_ids = [c.train_id for c in self.classes]
        self.color_to_train_ids = dict(zip(self.colors,self.train_ids))
        self.train_ids_to_color = dict(zip(self.train_ids,self.colors))

    
        self.root_dir = root_dir
        self.image_dir = 'translation_random_views/random_views'
        self.mask_dir = 'segmentation_random_views/random_views'
        self.depth_map_dir = 'depth_random_views/random_views'
        
        self.transform_x = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5])
        ])
        self.transform_y = transforms.Compose([
            transforms.ToTensor()
        ])

        self.transform_z = transforms.Compose([          
            transforms.ToTensor()
        ])
      
       
        self.image_files = list()
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(self.root_dir, self.image_dir)):
            self.image_files += [os.path.join(dirpath, file) for file in filenames if file.endswith('.png')]
        self.image_files.sort()

        self.mask_files = list()
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(self.root_dir, self.mask_dir)):
            self.mask_files += [os.path.join(dirpath, file) for file in filenames if file.endswith('.png')]
        self.mask_files.sort()


        self.depth_files = list()
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(self.root_dir, self.depth_map_dir)):
            self.depth_files += [os.path.join(dirpath, file) for file in filenames if file.endswith('.exr')]
        self.depth_files.sort()


    def encode_target(self, target):

        c,h,w = target.shape
        new_mask = torch.empty(h, w, dtype=torch.long)
        for k in self.color_to_train_ids:
            # Get all indices for current class
            idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)  # Check that all channels match
            new_mask[validx] = torch.tensor(self.color_to_train_ids[k], dtype=torch.long)
        new_mask = torch.unsqueeze(new_mask, dim=0)
        return new_mask

  
    def decode_target(self, target):
        c,h,w = target.shape
        
        new_mask = torch.zeros(h, w, 3, dtype=torch.uint8)
        for k in self.train_ids_to_color:
            # Get all indices for current class
            idx = (target==torch.tensor(k, dtype=torch.long))
            validx = (idx.sum(0) == 1)  # Check that all channels match
            new_mask[validx] = torch.tensor(self.train_ids_to_color[k], dtype=torch.uint8)

        # new_mask = new_mask.permute(2,0,1)
        return new_mask


        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_fn = os.path.join(self.root_dir, self.image_dir, self.image_files[idx])
        mask_fn = os.path.join(self.root_dir, self.mask_dir, self.image_files[idx])

        image = Image.open(self.image_files[idx]).convert('RGB')
        mask = Image.open(self.mask_files[idx]).convert('RGB')
        depth = readEXR(self.depth_files[idx])
    
        image, mask, depth = preprocess(image, mask, depth, transform= self.transform_x, target_transform= self.transform_y, depth_transform = self.transform_z , scale = (0.5,0.5), flip= True)
        
        mask = self.encode_target(mask)
        
        return image, mask, depth


def preprocess(image, mask, depth= None, transform = None, target_transform = None, depth_transform = None, flip=False, resize = None, scale=None, crop=None):
    if flip:
        if random.random() < 0.9:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if depth is not None:
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    if scale:
        w, h = image.size
        rand_log_scale = math.log(scale[0], 2) + random.random() * (math.log(scale[1], 2) - math.log(scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
        image = image.resize(new_size, Image.ANTIALIAS)
        mask = mask.resize(new_size, Image.NEAREST)
        if depth is not None:
            depth = depth.resize(new_size, Image.NEAREST)

    if resize:
        image = image.resize((resize,resize), Image.ANTIALIAS)
        mask = mask.resize((resize,resize), Image.NEAREST)
        
        if depth is not None:
            depth = depth.resize((resize,resize), Image.NEAREST)


    if transform:
        image = transform(image)
    if target_transform:
        mask = 255 * target_transform(mask)

    if depth is not None:
        if depth_transform:
            depth = target_transform(depth)

    mask = mask.long()
    
    if crop:
        h, w = image.shape[1], image.shape[2]
        pad_tb = max(0, crop[0] - h)
        pad_lr = max(0, crop[1] - w)
        image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
        mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)
        if depth is not None:
            depth = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(depth)  

        h, w = image.shape[1], image.shape[2]
        i = random.randint(0, h - crop[0])
        j = random.randint(0, w - crop[1])
        image = image[:, i:i + crop[0], j:j + crop[1]]
        mask = mask[i:i + crop[0], j:j + crop[1]]
        if depth is not None:
            depth = depth[i:i + crop[0], j:j + crop[1]]  

    if depth is not None:
        return image, mask, depth
    else:
        return image, mask