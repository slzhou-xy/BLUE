import os
from loguru import logger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import get_optimizer, get_scheduler, save_checkpoint
from datasets import get_dataloader
from model.network import Net

from downstream.downstream_utils import multi_cls_evaluation, binary_cls_evaluation


class Classifier(nn.Module):
    def __init__(self, args, model):
        super(Classifier, self).__init__()
        self.model = model
        self.mlp = nn.Sequential(
            nn.Linear(args.d_model, args.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(args.d_model, args.n_classes)
        )

    def forward(self, data):
        x = self.model.forward_encoder(data)
        return self.mlp(x)


@torch.no_grad()
def test_fn(model, data_loader, args):
    model.eval()
    pbar = tqdm(data_loader, ncols=100, desc='Test ')
    preds_list = []
    labels_list = []
    for batch_data, batch_label in pbar:
        for k, v in batch_data.items():
            batch_data[k] = v.cuda(non_blocking=True)
        preds = model(batch_data)
        preds_list.append(F.log_softmax(preds, dim=-1).detach().cpu().numpy())
        labels_list.append(batch_label.numpy())
    if args.dataset_name.startswith('porto'):
        results = multi_cls_evaluation(preds_list, labels_list, args.n_classes)
    else:
        results = binary_cls_evaluation(preds_list, labels_list)

    return results


@torch.no_grad()
def eval_fn(ep, model, data_loader, args):
    pbar = tqdm(data_loader, ncols=100, desc='Eval ')
    model.eval()
    eval_epoch_loss = []
    eval_preds_list = []
    eval_labels_list = []
    for batch_data, labels in pbar:
        for k, v in batch_data.items():
            batch_data[k] = v.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        preds = model(batch_data)
        loss = F.cross_entropy(preds, labels)
        eval_epoch_loss.append(loss.item())

        eval_preds_list.append(F.log_softmax(preds, dim=-1).detach().cpu().numpy())
        eval_labels_list.append(labels.detach().cpu().numpy())
        pbar.set_description(f"Epoch: {ep} | Eval  loss: {loss:.8f}")
    pbar.close()

    if args.dataset_name.startswith('porto'):
        eval_results = multi_cls_evaluation(eval_preds_list, eval_labels_list, args.n_classes)
    else:
        eval_results = binary_cls_evaluation(eval_preds_list, eval_labels_list)

    return np.mean(np.array(eval_epoch_loss)), eval_results


def train(args):
    pretrained_model_dir = os.path.join(args.save_dir, args.exp_id, 'ckpts')
    log_dir = os.path.join(args.save_dir, args.exp_id, 'task', args.task)
    model_save_dir = os.path.join(args.save_dir, args.exp_id, 'task', args.task)
    tsbd_dir = os.path.join(args.save_dir, args.exp_id, 'task', args.task)

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tsbd_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tsbd_dir)

    logger.add(sink=f'{log_dir}/{args.task}.log', mode='w',
               format="[<red>{time:YYYY-M-D HH:mm:ss}</red>] [{module}:{line} {function}()] -> {message}")
    logger.info(args)

    model = Net(args=args)

    load_path = os.path.join(pretrained_model_dir, f"model_{args.model_id}.pt")
    print("loading model at", load_path)
    model.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True)['model'])

    if args.probe_linear is True:
        for name, param in model.named_parameters():
            param.requires_grad = False

    cls_model = Classifier(args, model).to(device=args.device, non_blocking=True)

    train_loader, eval_loader, test_loader = get_dataloader(args, args.task)

    optim = get_optimizer(model.parameters(), args)
    sched = get_scheduler(optimizer=optim, args=args)

    best_epoch = -1
    best_loss = 1e10

    for ep in range(args.n_epochs):
        pbar = tqdm(train_loader, ncols=100, desc='Train')
        cls_model.train()
        sched.step(ep)
        train_epoch_loss = []
        train_preds_list = []
        train_labels_list = []
        for batch_data, batch_label in pbar:
            for k, v in batch_data.items():
                batch_data[k] = v.cuda(non_blocking=True)
            batch_label = batch_label.cuda(non_blocking=True)

            optim.zero_grad()
            preds = cls_model(batch_data)
            loss = F.cross_entropy(preds, batch_label)
            loss.backward()
            optim.step()

            with torch.no_grad():
                train_epoch_loss.append(loss.item())
                train_preds_list.append(F.log_softmax(preds, dim=-1).detach().cpu().numpy())
                train_labels_list.append(batch_label.detach().cpu().numpy())

            pbar.set_description(f"Epoch: {ep} | Train loss: {loss:.8f}")
        pbar.close()

        train_epoch_loss = np.mean(np.array(train_epoch_loss))

        if args.dataset_name.startswith('porto'):
            train_results = multi_cls_evaluation(train_preds_list, train_labels_list, args.n_classes)
        else:
            train_results = binary_cls_evaluation(train_preds_list, train_labels_list)

        eval_epoch_loss, eval_results = eval_fn(ep, cls_model, eval_loader, args)

        epoch_lr = optim.state_dict()['param_groups'][0]['lr']
        logger.info(f'Epoch: {ep:<2} | lr: {epoch_lr:8f}')
        logger.info(f"Train loss: {train_epoch_loss:.8f} | train results: {train_results}")
        logger.info(f"Eval  loss: { eval_epoch_loss:.8f} | eval  results: { eval_results}")

        writer.add_scalar('lr', epoch_lr, ep)
        writer.add_scalar('train loss', train_epoch_loss, ep)
        writer.add_scalar('eval loss', eval_epoch_loss, ep)

        save_path = os.path.join(model_save_dir, f"model_{ep}.pt")
        # save_checkpoint(save_path, cls_model, model_save_dir, sched)
        save_checkpoint(save_path, cls_model, model_save_dir, None)
        if eval_epoch_loss <= best_loss:
            best_loss = eval_epoch_loss
            best_epoch = ep
            patience = 0
        else:
            patience += 1
        if patience == args.patience:
            logger.info(f"Early stopping at epoch {ep} with loss {eval_epoch_loss}")
            break

     # * test
    load_path = os.path.join(model_save_dir, f"model_{best_epoch}.pt")
    print("loading model at", load_path)
    cls_model.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True)['model'])
    cls_model.to(device=args.device, non_blocking=True)

    test_results = test_fn(cls_model, test_loader, args)
    logger.info(f"Test  results: {test_results}")
