import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import nonechucks as nc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms as trn

_binary = np.vectorize(lambda x:1 if x >1 else 0)
  

class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgList, targetList, transform = None):
     
        self.imgList = imgList
        self.targetList = targetList
        self.transform = transform
            
    def __getitem__(self, index):

        img_path = self.imgList[index]
        image = Image.open(img_path).convert('RGB')
        label = np.array(Image.open(self.targetList[index]))[:,:,0]
    
        if self.transform:
            image = self.transform(image)
            label = torch.tensor(_binary(label),dtype=torch.float)
        sample = (image,label)
        
        return sample
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imgList) 
    
class EvaluateDataset(torch.utils.data.Dataset):
    def __init__(self, imgList, imgindexList, transform = None):
     
        self.imgList = imgList
        self.transform = transform
        self.imgindexList = imgindexList
        
    def __getitem__(self, index):

        img_path = self.imgList[index]
        img_index = self.imgindexList[index]
        
        image = Image.open(img_path).convert('RGB')
    
        if self.transform:
            image = self.transform(image)
        
        return image,img_path, img_index
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imgList) 
    
def accuracy(preds, label ):
    preds = torch.where(preds > 0.5, torch.ones( preds.shape).cuda(), torch.zeros( preds.shape).cuda()).squeeze()
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum().item()
    valid_sum = valid.sum().item()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum
   

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lr_deGamma ** (epoch // args.lr_deStep))
 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, folder = './', filename='checkpoint.pth.tar'):
    save_path = os.path.join(folder, filename)
    torch.save(state,  save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(folder, 'model_best.pth.tar')  )    