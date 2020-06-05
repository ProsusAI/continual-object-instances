from collections import defaultdict
import torch
import numpy as np
import random

from utils import make_directory


class MetricSampler:
    '''
    Base class for sampling positive and negative samples
    '''
    def __init__(self, train_data):
        self.train_data = train_data
        self.class_idxs, self.classes_list = self.__get_class_idxs()

    def __get_class_idxs(self):
        class_idxs = defaultdict(list)
        for idx, data_point in enumerate(self.train_data.data_files):
            target = self.train_data.map_to_target(data_point)
            if type(target) is int:
                class_idxs[target].append(idx)
            else:
                class_idxs[target.item()].append(idx)
        return class_idxs, list(class_idxs.keys())

    def sample_data(self):
        raise NotImplementedError


class TripletSampler(MetricSampler):
    '''
    samples triplets - one positive and negatives
    Input:
        train_data - object of dataset class
        neg_samples - number of negative samples 
    '''
    def __init__(self, train_data, neg_samples):
        super().__init__(train_data)
        self.neg_samples = neg_samples

    @property
    def is_triplet(self):
        return True

    def sample_data(self, anchor_id, anchor_target):
        pos_id = random.sample(self.class_idxs[anchor_target], k=1)[0]
        neg_ids = []
        for neg_sample in range(self.neg_samples):
            neg_class = random.choice(
                [x for x in self.classes_list if x != anchor_target])
            neg_id = random.sample(self.class_idxs[neg_class], k=1)[0]
            neg_ids.append(neg_id)

        return pos_id, neg_ids
