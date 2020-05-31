import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import device, args

from loss_utils import *


class TripletLoss(nn.Module):
    def __init__(self, task_method, neg_samples, temperature=1, margin=1.0):
        super(TripletLoss, self).__init__()
        self.device = device
        self.task_method = task_method

        if task_method == 'regression':
            self.triplet_criterion = TripletAdapt(margin)
        elif task_method == "classification":
            self.neg_samples = neg_samples
            self.temperature = temperature
            self.triplet_criterion = TripletCEAdapt(device, self.neg_samples)
            self.ce_criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, **kwargs):
        anchor = kwargs['anchor']
        pos = kwargs['pos']
        neg = kwargs['neg']
        if self.task_method == 'regression':
            losses, active_samples = self.triplet_criterion(anchor, pos, neg)
            return losses, active_samples
        elif self.task_method == "classification":
            shuffled_preds, targets = self.triplet_criterion(anchor, pos, neg)
            losses = self.ce_criterion(
                shuffled_preds/self.temperature, targets)
            correct = accuracy(shuffled_preds, targets)
            return losses.mean(), correct
