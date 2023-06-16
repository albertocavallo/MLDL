import os
import random

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from ptflops import get_model_complexity_info
from model import ENet
from utils import *
from timer import Timer
import pdb
import sys

sys.path.append("../..")

from config import cfg
from loading_data import loading_data


exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()

def main():

    cfg_file = open('../../config.py',"r")
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID)==1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    net = []   
    
    if cfg.TRAIN.STAGE=='all':
        net = ENet(only_encode=False)
        if cfg.TRAIN.PRETRAINED_ENCODER != '':
            encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
            del encoder_weight['classifier.bias']
            del encoder_weight['classifier.weight']
            # pdb.set_trace()
            net.encoder.load_state_dict(encoder_weight)
    elif cfg.TRAIN.STAGE =='encoder':
        net = ENet(only_encode=True)

    if len(cfg.TRAIN.GPU_ID)>1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net=net.cuda()

    net.train()
    #criterion = torch.nn.BCEWithLogitsLoss().cuda() # Binary Classification
    criterion = torch.nn.CrossEntropyLoss().cuda() # Multi Classification
    
    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    _t = {'train time' : Timer(),'val time' : Timer()} 

    for epoch in range(cfg.TRAIN.MAX_EPOCH):

        print(f"EPOCH {epoch+1}")
        
        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].toc(average=False)
        print('training time of one epoch: {:.2f}s'.format(_t['train time'].diff))
        _t['val time'].tic()
        
        #validate function
        validate_instanceSeg(val_loader, net, criterion, optimizer, epoch, restore_transform)
        
        _t['val time'].toc(average=False)
        print('val time of one epoch: {:.2f}s'.format(_t['val time'].diff))

def train(train_loader, net, criterion, optimizer, epoch):


    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
   
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels.long())

        loss.backward()
        optimizer.step()
        

def validate_instanceSeg(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    
    num_classes = 5
    iou_sum_classes = [0,0,0,0,0]
    
    for vi, data in enumerate(val_loader, 0):        

        inputs, labels = data

        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        
        outputs = net(inputs)

        #multi-classification --> softmax
        outputs = F.softmax(outputs, dim=1)

        for c in range (num_classes):

          #prediction mask  
          pred_mask = (outputs.argmax(dim=1) == c).cpu().numpy()

          #labels mask
          labels_mask = (labels == c).cpu().numpy()

          #computation of the iou for that class using the function in utils.py
          class_iou = calculate_mean_iu(pred_mask, labels_mask, 5)

          #accumulate the values in an array
          iou_sum_classes[c] += class_iou

    #dividing each value for len(val_loader)
    mean_iu_classes = [x / len(val_loader) for x in iou_sum_classes] 

    print(f"MEAN IOU - NOTHING (0): {mean_iu_classes[0]}")
    print(f"MEAN IOU - ALU     (1): {mean_iu_classes[1]}")
    print(f"MEAN IOU - CARTON  (2): {mean_iu_classes[2]}")
    print(f"MEAN IOU - BOTTLE  (3): {mean_iu_classes[3]}")
    print(f"MEAN IOU - NYLON   (4): {mean_iu_classes[4]}")
    
    
    net.train()
    criterion.cuda()


if __name__ == '__main__':
    main()


