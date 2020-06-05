import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict

from utils import make_directory, args


class Cars3D:
    '''
    Pre-processing class for Cars3D dataset
    Inputs:
        root - relative path to data folder
        mode - train, query or gallery set
        train_size - number of cars in train set with all its instances
        image_size - train image size
        query_split - number of instances per car in query. The remaining are part of the gallery set
    '''
    def __init__(self, root, mode, train_size=100, image_size=32, query_split=10):
        self.data_path = os.path.join(root, "Cars3D", "images")
        self.mode = mode
        self.train = True if mode == "train" else False
        self.train_size = train_size
        self.query_split = query_split

        self.tensor_transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()])

        self.data_files = self.read_data()

    def read_data(self):
        cars = [f for f in os.listdir(self.data_path) if f.startswith("car")]
        cars.sort()
        self.map_car2idx(cars)
        if self.train:
            cars = cars[:self.train_size]
        else:
            cars = cars[self.train_size:]

        return self.read_all_files(cars)

    def read_all_files(self, paths):
        all_cars = []
        for path in paths:
            temp_path = os.path.join(self.data_path, path)
            cars = [os.path.join(temp_path, f) for f in os.listdir(
                os.path.join(temp_path)) if f.startswith("car")]
            if self.mode == "train":
                all_cars += cars
            elif self.mode == "query":
                all_cars += cars[:self.query_split]
            elif self.mode == "gallery":
                all_cars += cars[self.query_split:]
        return all_cars

    def map_car2idx(self, cars):
        self.car2idx = {}
        self.idx2car = {}
        for idx, car in enumerate(cars):
            self.car2idx[car] = idx
            self.idx2car[idx] = car

    def map_to_target(self, data_point):
        target = self.car2idx[data_point.split(
            "/")[-2]]
        return target

    def load_data(self, data_point):
        image = Image.open(data_point)
        image = self.process_data(image)
        target = self.map_to_target(data_point)
        return image.squeeze(), target

    def process_data(self, image):
        image = self.tensor_transform(image)
        return image.unsqueeze(dim=0)

    def set_datafiles(self, data_files):
        self.data_files = data_files

    def __getitem__(self, idx):
        return self.load_data(self.data_files[idx])

    def __len__(self):
        return len(self.data_files)
