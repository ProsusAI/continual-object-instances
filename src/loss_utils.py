import torch
import torch.nn.functional as F
import torch.nn as nn

from utils import device, args, send_to_device


def accuracy(predictions, targets):
    predicted = torch.argmax(predictions, 1)
    correct = (predicted == targets).sum().item()
    return correct


class TripletAdapt(nn.Module):
    """
    From: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample

    """
    def __init__(self, margin=1.0):
        super(TripletAdapt, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean(), (losses > 0).sum().item()


class TripletCEAdapt(nn.Module):
    """
    Triplet loss adaptor for Cross-Entropy
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    Returns predictions and targets (positive or negative indicator)
    """
    def __init__(self, device, neg_samples):
        super(TripletCEAdapt, self).__init__()
        self.device = device
        self.neg_samples = neg_samples

    def forward(self, anchor, pos, neg):
        batch_size = anchor.size()[0]

        # dot product between vectors
        anchor_pos = torch.einsum("bi,bi -> b", [anchor, pos])
        rep_anchor = anchor.repeat_interleave(
            torch.tensor([self.neg_samples]).to(device), dim=0)
        anchor_neg = torch.einsum("bi,bi -> b", [rep_anchor.to(device), neg])

        # Put predictions together
        anchor_neg = anchor_neg.view(-1, self.neg_samples)
        predictions = torch.cat((anchor_pos.unsqueeze(1), anchor_neg), dim=1)

        # shuffle predictions
        random_pos_idx = torch.stack(
            [torch.randperm(self.neg_samples+1) for _ in range(batch_size)])
        random_pos_idx = random_pos_idx.to(device)
        shuffled_preds = torch.gather(predictions, 1, random_pos_idx)

        # Generate Targets
        targets = (random_pos_idx == 0).nonzero(as_tuple=True)[
            1]  # 0 because we concatenate with pos in id = 0
        targets = targets.to(device)

        return shuffled_preds, targets


def knowledge_distillation(outputs, targets, exp=1, size_average=True, eps=1e-5):
    '''
    Adapted from: https://github.com/joansj/hat
     '''
    out = torch.nn.functional.softmax(outputs, dim=1)
    tar = torch.nn.functional.softmax(targets, dim=1)

    if exp != 1:
        out = out.pow(exp)
        out = out/out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar/tar.sum(1).view(-1, 1).expand_as(tar)
    out = out+eps/out.size(1)
    out = out/out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar*out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce
