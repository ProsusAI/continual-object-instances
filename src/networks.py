import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils import device, args
from utils import make_directory

TRANSFER_MODEL_PATH = make_directory("../transfer_models/")


class LeNetEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=32):
        super(LeNetEmbeddingNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5),
                                     self.batchnorm1,
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5),
                                     self.batchnorm2,
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 128, 2),
                                     self.batchnorm3,
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2)
                                     )

        self.fc = nn.Sequential(nn.Linear(512, 256),
                                self.batchnorm4,
                                nn.ReLU(),
                                nn.Linear(256, 64),
                                nn.ReLU(),
                                nn.Linear(64, self.embedding_dim)
                                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        if args.normalize:
            return F.normalize(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ResNetEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=2):
        super(ResNetEmbeddingNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Linear(512, self.embedding_dim)

    def forward(self, x):
        output = self.resnet18(x)
        if args.normalize:
            return F.normalize(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def change_embedding_dim(self, emb_dim):
        self.embedding_dim = emb_dim
        self.backbone_net.embedding_dim = emb_dim


class TripletNet(nn.Module):
    def __init__(self, backbone_net):
        super(TripletNet, self).__init__()
        self.backbone_net = backbone_net
        self.embedding_dim = backbone_net.embedding_dim

    def forward(self, anchor, pos, neg):
        anchor = self.backbone_net(anchor)
        pos = self.backbone_net(pos)
        neg = self.backbone_net(neg)
        return anchor, pos, neg

    def get_embedding(self, x):
        return self.backbone_net(x)

    def change_embedding_dim(self, emb_dim):
        self.embedding_dim = emb_dim
        self.backbone_net.embedding_dim = emb_dim


def initialize_model(model_name, embedding_dim):
    if model_name == "lenet":
        backbone_net = LeNetEmbeddingNet(embedding_dim)
    elif model_name == "resnet":
        backbone_net = ResNetEmbeddingNet(embedding_dim)
    else:
        raise ValueError("Model not defined")
    return TripletNet(backbone_net).to(device)


def freeze_conv_layers(model):
    if args.model == "lenet":
        model = freeze_lenet_conv_layers(model)
    elif args.model == "resnet":
        model = freeze_resnet_conv_layers(model)
    else:
        raise ValueError("Model not defined")
    return model


def freeze_lenet_conv_layers(model):
    for param in model.backbone_net.convnet.parameters():
        param.requires_grad = False
    return model


def freeze_resnet_conv_layers(model):
    for name, param in model.backbone_net.named_parameters():
        if "layer4" in name or "fc" in name:
            pass
        else:
            param.requires_grad = False
    return model


def freeze_layers(old_model, model):
    if args.continuous_learning_method == "finetune":
        model = freeze_conv_layers(model)
    elif args.continuous_learning_method == "lfl":
        model = freeze_conv_layers(model)
        for param in old_model.parameters():
            param.requires_grad = False
    if args.freeze:
        if args.continuous_learning_method != "naive":
            model = freeze_conv_layers(model)
    return old_model, model
