import os
import random
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from ptflops import get_model_complexity_info
from model_B import BiSeNet
from config import cfg
from loading_data import loading_data
from utils import *
from timer import Timer
import pdb
import torch.nn.functional as F

exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()


def bce_loss(pred, label):
    """Computes binary cross-entropy loss for a binary classification task.

    Args:
        pred (torch.Tensor): Predicted logits (before sigmoid) of shape (batch_size, 1, H, W).
        label (torch.Tensor): Ground-truth binary labels of shape (batch_size, H, W).

    Returns:
        torch.Tensor: Scalar tensor of mean binary cross-entropy loss.
    """
    # Flatten tensors
    pred = pred.view(-1)
    label = label.view(-1)

    # Compute binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), reduction='mean')

    return loss


def main():

    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID)==1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    
    net = BiSeNet(5, 'resnet18')

    if len(cfg.TRAIN.GPU_ID)>1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net=net.cuda()

    net.train()
    criterion = torch.nn.CrossEntropyLoss().cuda() # Binary Classification
    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    _t = {'train time' : Timer(),'val time' : Timer()} 
    validate(val_loader, net, criterion, optimizer, -1, restore_transform)
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].toc(average=False)
        print('training time of one epoch: {:.2f}s'.format(_t['train time'].diff))
        _t['val time'].tic()
        validate(val_loader, net, criterion, optimizer, epoch, restore_transform)
        _t['val time'].toc(average=False)
        print('val time of one epoch: {:.2f}s'.format(_t['val time'].diff))
'''
    #computing flops and number of parameters
    flops, num_parameters = get_model_complexity_info(net, (3,800,800), as_strings=True)
    print(flops, num_parameters)
    
    #model size = (num_parameters*4/1024)/1024 --> size in MB
    model_size = (int(num_parameters)*4/1024)/1024
    print("Model size: " + str(model_size) + " MB")
'''
def train(train_loader, net, criterion, optimizer, epoch):
    print('OH SI')
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
   
        optimizer.zero_grad()
        outputs1, outputs2, outputs3= net(inputs)

        loss1=loss2=loss3 = 0

        for i in range(5):
          loss1 += criterion(outputs1[:,i,:,:], labels.float())
          loss2 += criterion(outputs2[:,i,:,:], labels.float())
          loss3 += criterion(outputs3[:,i,:,:], labels.float())

        loss =(loss1+loss2+loss3)/5
        
        loss.backward()
        
        
        optimizer.step()


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    
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
          pred_mask = (outputs.argmax(dim=1) == c).cpu().numpy()
          labels_mask = (labels == c).cpu().numpy()
          class_iou = calculate_mean_iu(pred_mask, labels_mask, 5)
          iou_sum_classes[c] += class_iou

    #dividing each value for len(val_loader)
    mean_iu_classes = [x / len(val_loader) for x in iou_sum_classes] 

    print("MEAN IOU:")
    print(mean_iu_classes)

    net.train()
    criterion.cuda()



if __name__ == '__main__':
    main()



