import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

from samplers import TripletSampler
from utils import args


class BaseData(Dataset):
    def __init__(self, data, neg_samples, sampling_method="triplet"):
        self.data = data
        self.sampler = TripletSampler(self.data, neg_samples)

        self.is_train = data.train
        self.data_length = len(data)
        self.n_groundtruths = self.groundtruths_per_class()
        self.is_triplet = self.sampler.is_triplet

    def groundtruths_per_class(self):
        n_groundtruths = dict()
        for class_id, class_idxs in self.sampler.class_idxs.items():
            n_groundtruths[class_id] = len(class_idxs)
        return n_groundtruths

    def __getitem__(self, idx):
        data_items = dict()
        anchor, anchor_target = self.data[idx]
        data_items["anchor"] = anchor
        data_items["anchor_target"] = anchor_target
        if self.is_train:
            data_items["pos"], data_items["neg"] = self.__getitem_triplet(
                idx, anchor_target)
            data_items["neg"] = torch.cat(data_items["neg"])

        return data_items

    def __getitem_triplet(self, idx, anchor_target):
        pos_id, neg_ids = self.sampler.sample_data(idx, anchor_target)
        pos, _ = self.data[pos_id]
        negs = [self.data[neg_id][0] for neg_id in neg_ids]
        return pos, negs

    def __len__(self):
        return self.data_length

    def show_image(self, idx):
        im = self.data.data[idx]
        trans = transforms.ToPILImage()
        im = trans(im)
        im.show()


def get_train_loader(train_partition):
    train_dataset = BaseData(train_partition, args.neg_samples)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    return train_loader


def get_test_loaders(query_data, gallery_data):
    query_dataset = BaseData(query_data, args.neg_samples)
    query_loader = DataLoader(
        query_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    gallery_dataset = BaseData(
        gallery_data, args.neg_samples)
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    return query_loader, gallery_loader
