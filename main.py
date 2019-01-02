import argparse
import os
import random
import shutil
import time
import datetime
import warnings
import sys

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
from PIL import Image
 
import tifffile as tiff
from argparse import Namespace

from nets.unet import * 
from nets.utils import *


date_today = str(datetime.datetime.now().date()) 

parser = argparse.ArgumentParser(description='PyTorch-Unet Binary Classification Training')

parser.add_argument('data', metavar='DIR',
                    help='dataframe path to dataset ' 'include: index path label split')
parser.add_argument('-a', '--arch', metavar='ARCH', default='unet',
                    help='model architecture')
parser.add_argument('--input-dim', default=3, type=int, metavar='N',
                    help='number of channels of input data (default: 3)') 
parser.add_argument('--output-dim', default=1, type=int, metavar='N',
                    help='number of channels of output data (class numbers) (default: 1)') 
parser.add_argument('-s', '--save-folder', default = './model', metavar='PATH', type=str,
                    help='folder path to save model files and log files (default: ./model)')
parser.add_argument('-m', '--save-prefix', default = date_today, metavar='PATH', type=str,
                    help='prefix string to specify version or task type  (default: [date])')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate',  action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-p', '--pretrained', dest='pretrained',  action='store_true',
                    help='use pre-trained model')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)', dest='lr')
parser.add_argument('--lr-deGamma', default=0.1, type=float,
                    metavar='LR', help='learning rate deGamma (default: 0.1)', )
parser.add_argument('--lr-deStep', default=50, type=float,
                    metavar='LR', help='initial learning rate deStep (default: 50)', )
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--gpu', default= '1,2,3,4', type=str,
                    help='GPU id(s) to use.'
                         'Format: a comma-delimited list')
best_acc = 0

def main():
    args = parser.parse_args()
    
    print args

    if len(args.gpu) is not 0:
        print 'You have chosen a specific GPU(s) : {}'.format(args.gpu)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
        
    if not os.path.exists(args.save_folder) :
        os.makedirs(args.save_folder)
        
    main_worker(args)
    
def main_worker(args):
    global best_acc
 
    model = UNet(n_channels = args.input_dim, n_classes= args.output_dim)
    model = torch.nn.DataParallel(model).cuda()
    
    criterion = torch.nn.BCELoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), 
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print("Current accuracy in validation: {}".format(checkpoint['best_acc']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    tf = torchvision.transforms.Compose([
        #trn.Resize(args.img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5914, 0.6051, 0.5975], [0.0083, 0.0062, 0.0059])
    ])
    
    if args.evaluate:
        evaluate( model, criterion, tf, args)
        return

    imageDF = pd.read_pickle(args.data)
    trainDF = imageDF[imageDF.split =='train']
    valDF = imageDF[imageDF.split =='val']

    trainDataset = Dataset(trainDF.path.tolist(), trainDF.label.tolist(), tf)
    valDataset = Dataset(valDF.path.tolist(), valDF.label.tolist(), tf)

    train_loader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size= args.batch_size, 
            shuffle=False,
            num_workers= args.workers, 
            pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            valDataset,
            batch_size= args.batch_size, 
            shuffle=False,
            num_workers= args.workers, 
            pin_memory=True) 
    
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_file_name =  args.save_prefix + '_checkpoint.pth.tar'

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_folder, save_file_name  )
        
def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter()
    acc = AverageMeter()    

    model.train()
    
    for i, (imgs, target) in enumerate(train_loader):
        imgs = imgs.cuda()
        target = target.cuda()    
        output = model(imgs)
        loss = criterion(output, target)
        thisAcc, _ = accuracy(output, target)

        losses.update(loss.item(), imgs.size(0))
        acc.update(thisAcc, imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    date_time = str(datetime.datetime.now())
    print('Epoch: [{0}][{1}/{2}]\t'
          'Time {date_time}\t'
          'Loss {loss.sum:.4f} ({loss.avg:.4f})\t'
          'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
           epoch, i, len(train_loader), date_time= date_time,
              loss=losses, acc = acc))    
    
def validate(val_loader, model, criterion, args):   
    losses = AverageMeter()
    acc = AverageMeter()  
    
    model.eval()
    
    with torch.no_grad():
        for i, (imgs, target) in enumerate(val_loader):
            imgs = imgs.cuda()
            target = target.cuda()

            # compute output
            output = model(imgs)
            loss = criterion(output, target)
            thisAcc, _ = accuracy(output, target)

            losses.update(loss.item(), imgs.size(0))
            acc.update(thisAcc, imgs.size(0))

        print('\t Val:\t'             
              'Loss {loss.sum:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                i, len(val_loader),  loss=losses, acc = acc)) 
        
    return acc.avg

def evaluate( model, criterion, transformer, args):
    
    model.eval()
    
    imageDF = pd.read_pickle(args.data)
    evaluateDF = imageDF[imageDF.split=='test']
    evaluateDF['predict'] = None
    valDataset = EvaluateDataset(evaluateDF.path.tolist(), evaluateDF.index.tolist(), transformer)

    val_loader = torch.utils.data.DataLoader(
            valDataset,
            batch_size= args.batch_size, 
            shuffle=False,
            num_workers= args.workers, 
            pin_memory=True)

    resultList = []
    labelList= []
    
    with torch.no_grad():
        for batch_idx, (input, paths, img_indexes) in enumerate(val_loader):

            input = input.cuda()
            output = model(input)

            output = output.detach().cpu().numpy().squeeze()
            batchLength= input.size(0)
            predictBatchList = []
            
            for i in range(batchLength):
                predictBatchList.append(output[i])

            batchDF = pd.DataFrame(dict(predict = predictBatchList), index= range(batchLength))
            batchDF['path'] = paths
            batchDF.index = img_indexes

            evaluateDF.update(batchDF)
    evaluationPath = os.path.join(args.save_folder, args.save_prefix +'_evaluate.p')
    evaluateDF.to_pickle(evaluationPath)
    print 'Evaluation pickle saved at: ', evaluationPath
    
if __name__ == '__main__':
    main()