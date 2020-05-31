import torch
from tqdm import tqdm

from utils import device, args
from utils import save_model, send_to_device, print_train_progress
from metrics import evaluation


def train(model, criterion, train_loader, query_loader, gallery_loader, optimizer, experiment_name):
    for epoch in range(args.n_epochs):
        train_loss, metric = train_epoch(
            model, criterion, optimizer, train_loader)
        print_train_progress(epoch, train_loss, metric)
        if epoch % args.print_every == 0:
            evaluation(model, query_loader, gallery_loader)
    save_model(model, experiment_name)


def continuous_train(old_model, model, criterion, train_loader, query_loader, gallery_loader, optimizer, experiment_name):
    for epoch in range(args.n_epochs):
        if args.continuous_learning_method == "naive":
            train_loss, metric = train_epoch(
                model, criterion, optimizer, train_loader)
        elif args.continuous_learning_method == "finetune":
            train_loss, metric = train_epoch(
                model, criterion, optimizer, train_loader)
        elif args.continuous_learning_method == "lfl":
            train_loss, metric = train_lfl_epoch(
                old_model, model, criterion, optimizer, train_loader)
        elif args.continuous_learning_method == "lwf":
            train_loss, metric = train_lfl_epoch(
                old_model, model, criterion, optimizer, train_loader)
        elif args.continuous_learning_method == "ewc":
            train_loss, metric = train_ewc_epoch(
                old_model, model, criterion, optimizer, train_loader)
        else:
            raise ValueError(
                "Provided Continual Learning method does not exist")
        print_train_progress(epoch, train_loss, metric)
    save_model(model, experiment_name)


def train_epoch(model, criterion, optimizer, dataloader):
    model.train()
    total_loss = 0
    total_metrics = 0
    for idx, data_items in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        data_items = send_to_device(data_items, device)

        b, c, h, w = data_items["neg"].size()
        data_items["neg"] = data_items["neg"].view(
            b*args.neg_samples, int(c/args.neg_samples), h, w)

        anchor, pos, neg = model(
            data_items["anchor"], data_items["pos"], data_items["neg"])
        loss, metric = criterion(
            anchor=anchor, pos=pos, neg=neg, targets=data_items["anchor_target"])
        total_loss += loss.item()
        total_metrics += metric
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

    total_loss /= len(dataloader)
    if args.task_method == "regression":
        metric = total_metrics/len(dataloader)
    else:
        metric = total_metrics/len(dataloader.dataset)
    return total_loss, metric


def train_lfl_epoch(old_model, model, criterion, optimizer, dataloader):
    old_model.eval()
    model.train()
    total_loss = 0
    total_metrics = 0
    for idx, data_items in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        data_items = send_to_device(data_items, device)

        b, c, h, w = data_items["neg"].size()
        data_items["neg"] = data_items["neg"].view(
            b*args.neg_samples, int(c/args.neg_samples), h, w)

        anchor, pos, neg = model(
            data_items["anchor"], data_items["pos"], data_items["neg"])
        with torch.no_grad():
            old_anchor = old_model.get_embedding(data_items["anchor"])
        loss, metric = criterion(old_anchor=old_anchor, anchor=anchor,
                                 pos=pos, neg=neg, targets=data_items["anchor_target"])

        total_loss += loss.item()
        loss.backward()
        total_metrics += metric

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

    total_loss /= len(dataloader)
    if args.task_method == "regression":
        metric = total_metrics/len(dataloader)
    else:
        metric = total_metrics/len(dataloader.dataset)
    return total_loss, metric


def train_ewc_epoch(old_model, model, criterion, optimizer, dataloader):
    old_model.eval()
    model.train()
    total_loss = 0
    total_metrics = 0

    criterion.update_models(old_model, model)
    criterion.update_fisher(dataloader)
    data = []
    for idx, data_items in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        data_items = send_to_device(data_items, device)

        b, c, h, w = data_items["neg"].size()
        data_items["neg"] = data_items["neg"].view(
            b*args.neg_samples, int(c/args.neg_samples), h, w)

        anchor, pos, neg = model(
            data_items["anchor"], data_items["pos"], data_items["neg"])
        loss, metric = criterion(
            anchor=anchor, pos=pos, neg=neg, targets=data_items["anchor_target"])

        total_loss += loss.item()
        loss.backward()
        total_metrics += metric

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        data.append(data_items)

    total_loss /= len(dataloader)
    if args.task_method == "regression":
        metric = total_metrics/len(dataloader)
    else:
        metric = total_metrics/len(dataloader.dataset)
    return total_loss, metric
