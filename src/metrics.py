import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from utils import get_dataset_embeddings, make_directory, args
import os
import pickle
from collections import OrderedDict


RESULTS_PATH = make_directory("../{}/".format(args.output))


def get_pairwise_distances(model, query_loader, gallery_loader, function='euclidean'):
    query_embeddings, query_targets = get_dataset_embeddings(
        model, query_loader)
    gallery_embeddings, gallery_targets = get_dataset_embeddings(
        model, gallery_loader)
    distances = cdist(query_embeddings, gallery_embeddings, metric=function)
    return distances, query_targets, gallery_targets


def is_hit(is_target):
    if np.sum(is_target) > 0:
        return 1
    return 0


def query_recall(is_target, n_groundtruths):
    recall = np.sum(is_target) / n_groundtruths
    return np.asscalar(recall)


def query_AP(is_target, n_groundtruths, k):
    normalizer = min(n_groundtruths, k)
    positives_counter = np.cumsum(is_target)
    positives_counter = positives_counter*is_target
    results_counter = np.arange(1, k+1)
    scores = positives_counter/results_counter
    average_precision = float(scores.sum()) / normalizer

    return average_precision


def evaluate_metrics(model, query_loader, gallery_loader, k=-1, function='euclidean'):
    if k == -1:
        k = gallery_loader.dataset.data_length
    distances, query_targets, gallery_targets = get_pairwise_distances(
        model, query_loader, gallery_loader, function)
    sorted_dists = np.argsort(distances, axis=1)[:, :k]
    groundtruths_per_class = gallery_loader.dataset.n_groundtruths

    n_queries = query_targets.shape[0]
    sum_recall = 0.0
    hits = 0.0
    sum_average_precision = 0.0

    for query_id in range(n_queries):
        query_target = query_targets[query_id]
        results_indexes = sorted_dists[query_id]
        results_targets = gallery_targets[results_indexes]
        is_target = (results_targets == query_target).astype(int)
        n_groundtruths = groundtruths_per_class[query_target]

        # if query_id == 0:
        #     query_loader.dataset.show_image(query_id)
        #     gallery_loader.dataset.show_image(results_indexes[0])
        #     print(results_indexes[0])
        #     raise NotImplementedError

        # recall
        sum_recall += query_recall(is_target, n_groundtruths)
        # hit
        hits += is_hit(is_target)
        # AP
        sum_average_precision += query_AP(is_target, n_groundtruths, k)

    return {"recall": sum_recall/n_queries, "hit": hits/n_queries, "map": sum_average_precision/n_queries}


def evaluation(model, query_loader, gallery_loader):
    k_1 = evaluate_metrics(model, query_loader, gallery_loader, k=1)
    k_5 = evaluate_metrics(model, query_loader, gallery_loader, k=5)
    k_50 = evaluate_metrics(model, query_loader, gallery_loader, k=50)
    k_100 = evaluate_metrics(model, query_loader, gallery_loader, k=100)
    k_gallery = evaluate_metrics(model, query_loader, gallery_loader)
    print("K=1 Recall:{} HIT:{} mAP:{}".format(
        k_1["recall"], k_1["hit"], k_1["map"]))
    print("K=5 Recall:{} HIT:{} mAP:{}".format(
        k_5["recall"], k_5["hit"], k_5["map"]))
    print("K=50 Recall:{} HIT:{} mAP:{}".format(
        k_50["recall"], k_50["hit"], k_50["map"]))
    print("K=100 Recall:{} HIT:{} mAP:{}".format(
        k_100["recall"], k_100["hit"], k_100["map"]))
    print("K=|gallery| Recall:{} HIT:{} mAP:{}".format(
        k_gallery["recall"], k_gallery["hit"], k_gallery["map"]))

    return k_1, k_5, k_50, k_100, k_gallery


def initialize_metrics():
    maps = OrderedDict(
        {"map_1": [], "map_5": [], "map_50": [], "map_100": [], "map_gallery": []})
    hits = OrderedDict(
        {"hit_1": [], "hit_5": [], "hit_50": [], "hit_100": [], "hit_gallery": []})
    recalls = OrderedDict(
        {"recall_1": [], "recall_5": [], "recall_50": [], "recall_100": [], "recall_gallery": []})
    return maps, hits, recalls


def update_metrics(ks, maps, hits, recalls):
    for k, key in zip(ks, maps.keys()):
        maps[key].append(k["map"])
    for k, key in zip(ks, hits.keys()):
        hits[key].append(k["hit"])
    for k, key in zip(ks, recalls.keys()):
        recalls[key].append(k["recall"])
    return maps, hits, recalls


def write_results(metrics, experiment_name):
    parent_dir = make_directory(RESULTS_PATH)
    for metric in metrics.keys():
        with open(os.path.join(parent_dir, experiment_name+"_"+metric)+'.pkl', 'wb') as f:
            pickle.dump(metrics[metric], f)
