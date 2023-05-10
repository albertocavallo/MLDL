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
from bisenet_v1 import BiSeNet
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


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        # intersect = (output * target).sum(-1).sum() + self.epsilon
        # denominator = ((output + target).sum(-1)).sum() + self.epsilon

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice
        # return 1 - 2. * intersect / denominator


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

    
    net = BiSeNet(1, 'resnet18')

    if len(cfg.TRAIN.GPU_ID)>1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net=net.cuda()

    net.train()
    criterion = torch.nn.BCEWithLogitsLoss().cuda() # Binary Classification
    #criterion=DiceLoss().cuda()
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
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
   
        optimizer.zero_grad()
        outputs1, outputs2, outputs3= net(inputs)

        loss1 = bce_loss(outputs1.squeeze(1), labels)
        loss2 = bce_loss(outputs2.squeeze(1), labels)
        loss3 = bce_loss(outputs3.squeeze(1), labels)
        loss =loss1 + loss2+ loss3
        #loss = criterion(outputs, labels.float().unsqueeze(1))
        
        #loss = criterion(torch.stack(list(outputs[2]), dim=0), torch.stack(list(labels.unsqueeze(1).float()),dim=0))
        loss.backward()
        
        
        optimizer.step()


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    iou_ = 0.0
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        outputs = net(inputs)
        #for binary classification
        outputs[outputs>0.5] = 1
        outputs[outputs<=0.5] = 0
        #for multi-classification ???

        iou_ += calculate_mean_iu([outputs.squeeze_(1).data.cpu().numpy()], [labels.data.cpu().numpy()], 2)
    mean_iu = iou_/len(val_loader)   

    print('[mean iu %.4f]' % (mean_iu)) 
    net.train()
    criterion.cuda()


if __name__ == '__main__':
    main()

