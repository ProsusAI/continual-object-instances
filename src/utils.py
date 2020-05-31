import os
import argparse
import torch
import numpy as np
import uuid
from collections import OrderedDict


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default="Cars3D",
                        help="dataset to process")
    parser.add_argument('-ds', '--data_splits', type=int, default=1,
                        help="number of equal data partitions")
    parser.add_argument('-s', '--sampling_method', default="triplet",
                        help="sampling method")
    parser.add_argument('-sp', '--split_method', default="incremental",
                        help="split method")
    parser.add_argument('-mo', '--model', default="lenet", choices=["lenet", "resnet"],
                        help="backbone model")
    parser.add_argument('-clm', '--continuous_learning_method', default="naive", choices=["naive", "finetune", "lfl", "lwf", "ewc"],
                        help="continual learning approach")
    parser.add_argument('-t', '--task_method', default="regression", choices=["regression", "classification"],
                        help="benchmark or NCE approach")
    parser.add_argument('-e', '--n_epochs', type=int, default=10,
                        help="define the number of epochs")
    parser.add_argument('--data_path', default="../../datasets/",
                        help="root data folder")
    parser.add_argument('-o', '--output', required=True,
                        help="output folder name folder")
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help="batch size")
    parser.add_argument('-l', '--lr', type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument('-lamb', '--lambda_lfl', type=float, default=0.05,
                        help="lfl weight in euclidean distance between anchors")
    parser.add_argument('-lamb_lwf', '--lambda_lwf', type=float, default=0.05,
                        help="lwf weight in knowledge distillation between anchors")
    parser.add_argument('-emb', '--embedding_dim', type=int, default=32,
                        help="embedding size")
    parser.add_argument('-im', '--image_size', type=int, default=32,
                        help="image size")                        
    parser.add_argument('-w', '--num_workers', type=int, default=1,
                        help="parallel workers")
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help="GPU ID")
    parser.add_argument('-p', '--print_every', type=int, default=100,
                        help="print steps")
    parser.add_argument('-n', '--neg_samples', type=int, default=9,
                        help="Number of negative samples for CE loss")
    parser.add_argument('-temp', '--temperature', type=float, default=1.0,
                        help="Temperature for softmax under the NCE setting")
    parser.add_argument('-lamb_ewc', '--lambda_ewc', type=float, default=0.05,
                        help="lambda ewc")
    parser.add_argument('--normalize', action='store_true', help= "normalize network outputs")
    parser.add_argument('--train_full', action='store_true', help="cumulative/offline training")
    parser.add_argument('--freeze', action='store_true', help="freeze conv layers, used in Naive approach")


    args = parser.parse_args()

    # overwrite number of negative samples
    if args.task_method == "classification":
        args.neg_samples = args.neg_samples
    else:
        args.neg_samples = 1

    return args


args = config()
device = torch.device("cuda:{}".format(args.gpu)
                      if torch.cuda.is_available() else "cpu")


def make_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def save_model(model, experiment_name):
    models_path = make_directory("../models/")
    torch.save(model.state_dict(), os.path.join(
        models_path, experiment_name+'.pt'))


def load_model(model, experiment_name, path="../models/"):
    model.load_state_dict(torch.load(
        os.path.join(path, experiment_name+'.pt'), map_location=torch.device(device)))
    return model


def send_to_device(data_items, device):
    for key in data_items.keys():
        data_items[key] = data_items[key].to(device)
    return data_items


@torch.no_grad()
def get_dataset_embeddings(model, dataloader):
    model.eval()
    embeddings_matrix = np.zeros(
        (len(dataloader.dataset), model.embedding_dim))
    targets_vector = np.zeros((len(dataloader.dataset)))
    k = 0
    for idx, data_items in enumerate(dataloader):
        n_datapoints = data_items["anchor_target"].size()[0]
        anchor_embeddings = model.get_embedding(
            data_items["anchor"].to(device))

        embeddings_matrix[k:k+n_datapoints,
                          :] = anchor_embeddings.cpu().numpy()
        targets_vector[k:k+n_datapoints] = data_items["anchor_target"].numpy()
        k += n_datapoints

    return embeddings_matrix, targets_vector


def get_experiment_name():
    experiment_id = str(uuid.uuid4().fields[-1])[:5]
    experiment_name = args.dataset + "_" + \
                    str(args.embedding_dim) + "_" + args.sampling_method + "_" + \
                    str(args.neg_samples) + "_" + args.task_method + "_" + args.model + \
                    "_" + args.continuous_learning_method + "_" + \
                    str(args.data_splits) + "_" + str(args.normalize) + "_" + str(args.temperature) + \
                    "_" + str(args.lr) + "_" + str(args.n_epochs) + "_" + \
                    str(args.lambda_ewc) + "_" + str(args.image_size)

    experiment_name += "_" + experiment_id
    return experiment_name


def get_partitions():
    partitions = OrderedDict(
        {str(i+1): 1/args.data_splits for i in range(args.data_splits)})
    partitions_tune = [{"trained": str(i), "tune": str(i+1)}
                       for i in range(1, args.data_splits)]

    partitions_train = []
    if args.train_full or args.data_splits == 1:
        partitions_train.append("full")
    if partitions_tune:
        partitions_train.append("1")
    return partitions, partitions_train, partitions_tune


def print_train_progress(epoch, train_loss, metric):
    if args.task_method == "classification":
        print("Epoch: {} Loss: {} Accuracy: {}".format(epoch, train_loss, metric))
    elif args.task_method == "regression":
        print("Epoch: {} Loss: {} Active samples: {}".format(
            epoch, train_loss, metric))
    else:
        raise ValueError("Unknown task_method: {}".format(args.task_method))
