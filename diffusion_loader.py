import math
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import random


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class DiffusionDataset(Dataset): 
    '''
    your folder should have following structrues:
    
    root
        --1.jpg
        --2.jpg
        --...
    '''
    def __init__(self,
                 root: str,
                 imgSize=[32,32],
                 max_step=2000):

        self.root = root
        self.width, self.height = imgSize
        self.imgList = os.listdir(root)
        self.list = os.listdir(root)
        self.max_step = max_step
    def __getitem__(self, index,t=None):
        index = random.randint(0,9)
        path = os.path.join(self.root, self.imgList[int(index)])
        img = pil_loader(path)
        img = F.to_tensor(img)
        img = F.resize(img,(self.width,self.height))
        img = img*2 -1
        if t is None:
            t = int(np.random.randint(self.max_step, size=1))
        alpha = 1 - math.sqrt((t+1)/self.max_step)
        noise = torch.normal(0,1,img.shape)
        noisy_img = math.sqrt(alpha)*img+math.sqrt(1-alpha)*noise
        return noisy_img,img,t

    def __len__(self):
        return len(self.imgList)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = DiffusionDataset("/mnt/d/Datasets/MiniImageNet/train",[64,64])
    for i in range(0,2000,20):
        print(i)
        img,noisy_img,t = dataset.__getitem__(0,i)
        img = (img + 1)/2
        noisy_img = (noisy_img + 1)/2
        img,noisy_img = F.to_pil_image(img),F.to_pil_image(noisy_img)
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(img)
        ax2.imshow(noisy_img)
        plt.show()