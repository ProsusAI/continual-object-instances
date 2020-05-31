from collections import defaultdict
import math
import random
import copy


class BaseInstanceSplitter:
    '''
    Class that takes in a dataset and split this into partitions. 
    Returns a dict names and partitioned datasets. The "full" key is reserved for the whole dataset
    '''

    def __init__(self, CarDataset, partitions):
        self.CarDataset = CarDataset
        self.original_data_files = self.CarDataset.data_files.copy()
        self.partition_names = list(partitions.keys())
        self.partition_ratios = list(partitions.values())

        if CarDataset.train:
            if sum(self.partition_ratios) > 1 or "full" in self.partition_names:
                raise ValueError("Invalid Partition")

        self.all_targets = self.get_all_targets()
        self.items_per_partition = self.get_items_per_partition()

    def split_data(self):
        datasets = {}
        self.CarDataset.set_datafiles(self.original_data_files)
        datasets["full"] = copy.deepcopy(self.CarDataset)  # benchmark

        for partition_id in self.partition_names:
            self.load_split_data(partition_id)
            datasets[partition_id] = copy.deepcopy(
                self.CarDataset)  # partitions
        return datasets

    def load_split_data(self, partition_id):
        selected_data_files = [self.original_data_files[image_id]
                               for image_id in self.ids_in_partition[partition_id]]
        self.CarDataset.set_datafiles(selected_data_files)

    def get_items_per_partition(self):
        self.class_idxs = self.get_class_idxs()
        items_per_partition = defaultdict(dict)
        for partition_id, partition_prob in zip(self.partition_names, self.partition_ratios):
            for target, class_ids in self.class_idxs.items():
                n_items = math.floor(len(class_ids)*partition_prob)
                items_per_partition[partition_id][target] = n_items
        return items_per_partition

    def get_all_targets(self):
        targets = []
        for data_point in self.CarDataset.data_files:
            targets.append(self.CarDataset.map_to_target(data_point))
        return targets

    def get_class_idxs(self):
        class_idxs = defaultdict(list)
        for idx, target in enumerate(self.all_targets):
            class_idxs[target].append(idx)
        return class_idxs

    def splitter(self):
        raise NotImplementedError


class IncrementalInstanceSplitter(BaseInstanceSplitter):
    '''
    Selects all the viewpoints for the selected classes incrementally. 
    Ensures that a class that was seen before, is never seen again.
    '''

    def __init__(self, CarDataset, partitions):
        super().__init__(CarDataset, partitions)
        self.ids_in_partition = self.splitter()

    def splitter(self):
        ids_in_partition = defaultdict(list)
        classes_in_partition = self.__get_classes_in_partition()
        all_classes = list(self.class_idxs)

        for partition_id in self.partition_names:
            selected_classes = random.sample(
                all_classes, k=classes_in_partition[partition_id])
            for selected_class in selected_classes:
                ids_in_partition[partition_id] += self.class_idxs[selected_class]

            all_classes = set(all_classes) - set(selected_classes)

        return ids_in_partition

    def __get_classes_in_partition(self):
        classes_in_partition = dict()
        for p_id, p_prob in zip(self.partition_names, self.partition_ratios):
            classes_in_partition[p_id] = math.floor(
                len(self.class_idxs)*p_prob)
        return classes_in_partition
