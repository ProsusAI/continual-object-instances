import torch
from copy import deepcopy

from data_splitters import *
from datasets import *
from data import get_train_loader, get_test_loaders

from losses import *
from continuous_losses import *

from networks import initialize_model, freeze_layers, freeze_conv_layers
from train import train, continuous_train
from metrics import initialize_metrics, evaluation, update_metrics, write_results

import utils
from utils import device

args = utils.config()
print(args)

experiment_name = utils.get_experiment_name()
data_path = utils.make_directory(args.data_path)

partitions, partitions_train, partitions_tune = utils.get_partitions()

if args.dataset == "Cars3D":
    train_data = Cars3D(root=args.data_path, mode="train",
                        image_size=args.image_size)
    query_data = Cars3D(root=args.data_path, mode="query",
                        image_size=args.image_size)
    gallery_data = Cars3D(root=args.data_path,
                          mode="gallery", image_size=args.image_size)
else:
    raise ValueError("Provided dataset does not exist")

if args.sampling_method == "triplet":
    criterion = TripletLoss(
        args.task_method, args.neg_samples, args.temperature)
else:
    raise ValueError("Provided sampling does not exist")

if args.split_method == "incremental":
    splitter = IncrementalInstanceSplitter(train_data, partitions=partitions)
else:
    raise ValueError("Provided split method does not exist")

train_partitions = splitter.split_data()
query_loader, gallery_loader = get_test_loaders(query_data, gallery_data)

metrics = initialize_metrics()

for p_id in partitions_train:
    model = initialize_model(args.model, args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = get_train_loader(train_partitions[p_id])

    model_name = experiment_name + "_{}".format(p_id)
    train(model, criterion, train_loader, query_loader, gallery_loader,
          optimizer, model_name)

    print(model_name)
    ks = evaluation(model, query_loader, gallery_loader)
    metrics = update_metrics(ks, *metrics)


if args.continuous_learning_method == "lfl":
    criterion = Triplet_LFL(triplet_criterion=criterion, lamb=args.lambda_lfl)
if args.continuous_learning_method == "lwf":
    criterion = Triplet_LWF(triplet_criterion=criterion, lamb=args.lambda_lwf)
if args.continuous_learning_method == "ewc":
    criterion = Triplet_EWC(triplet_criterion=criterion, lamb=args.lambda_ewc)


# Fine-Tune
if partitions_tune:
    for idx, continuous_set in enumerate(partitions_tune):
        model = initialize_model(args.model, args.embedding_dim)
        model = utils.load_model(model, experiment_name +
                                 "_{}".format(continuous_set["trained"]))
        old_model = deepcopy(model)

        old_model, model = freeze_layers(old_model, model)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        train_loader = get_train_loader(
            train_partitions[continuous_set["tune"]])

        model_name = experiment_name + "_{}".format(continuous_set["tune"])
        if idx == 0 and args.continuous_learning_method == "ewc":
            criterion.initialize_fisher(train_loader, old_model, model)
        continuous_train(old_model, model, criterion, train_loader, query_loader, gallery_loader, optimizer,
                         model_name)

        print(model_name)
        ks = evaluation(model, query_loader, gallery_loader)
        metrics = update_metrics(ks, *metrics)

maps, hits, recalls = metrics
write_results({"map": maps, "hits": hits, "recalls": recalls}, experiment_name)
