import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional

class FocalLoss(nn.modules.loss._WeightedLoss):
    """
    根据如下代码修改得来
    https://github.com/gokulprasadthekkel/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
    """
    def __init__(self, weight=None, gamma=2, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
        self.ignore_index = ignore_index
    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,
                                 reduction=self.reduction,
                                 weight=self.weight, ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        return focal_loss
