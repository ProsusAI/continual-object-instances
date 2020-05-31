import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import device, args

from loss_utils import *
from losses import *


class Triplet_LFL(nn.Module):
    def __init__(self, triplet_criterion, lamb):
        super(Triplet_LFL, self).__init__()
        self.lamb = lamb
        self.triplet_criterion = triplet_criterion

    def forward(self, **kwargs):
        old_anchor = kwargs['old_anchor']
        anchor = kwargs['anchor']
        # triplet loss
        triplet_loss, metric = self.triplet_criterion(**kwargs)
        # Embedding loss
        if old_anchor is not None:
            loss_dist = torch.sum((old_anchor-anchor).pow(2))/2
        loss = triplet_loss + self.lamb*loss_dist

        return loss, metric


class Triplet_LWF(nn.Module):
    def __init__(self, triplet_criterion, lamb, T=2):
        super(Triplet_LWF, self).__init__()
        self.lamb = lamb
        self.T = T
        if args.task_method == 'regression':
            raise ValueError("Regression not defined for LWF")
        self.triplet_criterion = triplet_criterion

    def forward(self, **kwargs):
        old_anchor = kwargs['old_anchor']
        anchor = kwargs['anchor']
        # triplet regression loss
        triplet_loss, metric = self.triplet_criterion(**kwargs)
        # distillation loss
        loss_dist = knowledge_distillation(anchor, old_anchor, exp=1/self.T)
        loss = triplet_loss + self.lamb*loss_dist

        return loss, metric


class Triplet_EWC(nn.Module):
    """
    Adapted from https://github.com/joansj/hat
    """

    def __init__(self, triplet_criterion, lamb=0.05):
        super(Triplet_EWC, self).__init__()
        self.lamb = lamb
        self.triplet_criterion = triplet_criterion

        self.fisher = None
        self.model = None
        self.old_model = None

    def forward(self, **kwargs):
        triplet_loss, metric = self.triplet_criterion(**kwargs)

        # Regularization for all previous tasks
        loss_reg = 0
        for (name, param), (_, param_old) in zip(self.model.named_parameters(), self.old_model.named_parameters()):
            loss_reg += torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

        loss = triplet_loss + self.lamb*loss_reg
        return loss, metric

    def update_models(self, old_model, model):
        self.old_model = old_model
        self.model = model

    def initialize_fisher(self, dataloader, old_model, model):
        self.update_models(old_model, model)
        self.fisher = self.initialize_fisher_matrix_diag(dataloader)

    def update_fisher(self, dataloader):
        fisher_old = {}
        for n, _ in self.model.named_parameters():
            fisher_old[n] = self.fisher[n].clone()
        self.fisher = self.fisher_matrix_diag(dataloader)

        for n, _ in self.model.named_parameters():
            self.fisher[n] = 0.5*(self.fisher[n]+fisher_old[n])

    def initialize_fisher_matrix_diag(self, data):
        # Init
        fisher = {}
        n_datapoints = len(data)*args.batch_size

        for n, p in self.model.named_parameters():
            fisher[n] = 0*p.data
        # Compute
        self.model.train()
        for idx, data_items in enumerate(data):
            # Forward and backward
            data_items = send_to_device(data_items, device)
            self.model.zero_grad()
            b, c, h, w = data_items["neg"].size()
            data_items["neg"] = data_items["neg"].view(
                b*args.neg_samples, int(c/args.neg_samples), h, w)
            anchor, pos, neg = self.model(
                data_items["anchor"], data_items["pos"], data_items["neg"])

            loss, metric = self.triplet_criterion(
                anchor=anchor, pos=pos, neg=neg, targets=data_items["anchor_target"])
            loss.backward()

            # Get gradients
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += b*p.grad.data.pow(2)
        # Mean
        for n, _ in self.model.named_parameters():
            fisher[n] = fisher[n]/n_datapoints
            fisher[n] = fisher[n].clone().detach()
        return fisher

    def fisher_matrix_diag(self, data):
        # Init
        fisher = {}
        n_datapoints = len(data)*args.batch_size

        for n, p in self.model.named_parameters():
            fisher[n] = 0*p.data
        # Compute
        self.model.train()
        for idx, data_items in enumerate(data):
            # Forward and backward
            data_items = send_to_device(data_items, device)
            self.model.zero_grad()
            b, c, h, w = data_items["neg"].size()
            data_items["neg"] = data_items["neg"].view(
                b*args.neg_samples, int(c/args.neg_samples), h, w)
            anchor, pos, neg = self.model(
                data_items["anchor"], data_items["pos"], data_items["neg"])

            loss, metric = self.forward(
                anchor=anchor, pos=pos, neg=neg, targets=data_items["anchor_target"])
            loss.backward()

            # Get gradients
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += b*p.grad.data.pow(2)
        # Mean
        for n, _ in self.model.named_parameters():
            fisher[n] = fisher[n]/n_datapoints
            fisher[n] = fisher[n].clone().detach()
        return fisher
