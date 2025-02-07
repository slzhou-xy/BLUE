import os
from loguru import logger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import get_optimizer, get_scheduler, save_checkpoint
from datasets import get_dataloader
from model.network import Net

from downstream.downstream_utils import travel_time_evaluation


class TravelTimeEvaluator(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(args.d_model, args.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(args.d_model, 1)
        )

    def forward(self, data):
        traj_emb = self.encoder.forward_encoder(data)
        pred = self.mlp(traj_emb)
        return pred


@torch.no_grad()
def test_fn(model, data_loader):
    pbar = tqdm(data_loader, ncols=100, desc="Test ")
    model.eval()
    preds_list = []
    label_list = []
    for batch_data, batch_label in pbar:
        for k, v in batch_data.items():
            batch_data[k] = v.cuda(non_blocking=True)
        preds = model(batch_data)
        preds_list.append(preds.detach().cpu().numpy())
        label_list.append(batch_label.detach().cpu().numpy())
    results = travel_time_evaluation(preds_list, label_list)
    return results


@torch.no_grad()
def eval_fn(ep, model, data_loader):
    pbar = tqdm(data_loader, ncols=100, desc='Eval ')
    model.eval()
    eval_epoch_loss = []
    eval_preds_list = []
    eval_labels_list = []
    for batch_data, batch_label in pbar:
        for k, v in batch_data.items():
            batch_data[k] = v.cuda(non_blocking=True)
        batch_label = batch_label.cuda(non_blocking=True)
        preds = model(batch_data)
        loss = F.mse_loss(preds, batch_label)
        eval_epoch_loss.append(loss.item())
        eval_preds_list.append(preds.detach().cpu().numpy())
        eval_labels_list.append(batch_label.detach().cpu().numpy())
        pbar.set_description(f"Epoch: {ep} | Eval  loss: {loss:.8f}")
    pbar.close()

    eval_epoch_loss = np.mean(np.array(eval_epoch_loss))
    eval_results = travel_time_evaluation(eval_preds_list, eval_labels_list)
    return eval_epoch_loss, eval_results


def train(args):
    pretrained_model_dir = os.path.join(args.save_dir, args.exp_id, 'ckpts')
    log_dir = os.path.join(args.save_dir, args.exp_id, 'task', args.task)
    model_save_dir = os.path.join(args.save_dir, args.exp_id, 'task', args.task, 'ckpts')
    tsbd_dir = os.path.join(args.save_dir, args.exp_id, 'task', args.task)

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tsbd_dir, exist_ok=True)

    logger.add(sink=f'{log_dir}/{args.task}.log', mode='w',
               format="[<red>{time:YYYY-M-D HH:mm:ss}</red>] [{module}:{line} {function}()] -> {message}")
    logger.info(args)

    model = Net(args=args)
    load_path = os.path.join(pretrained_model_dir, f"model_{args.model_id}.pt")
    print("loading model at", load_path)
    model.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True)['model'])

    train_loader, eval_loader, test_loader = get_dataloader(args, args.task)

    if args.probe_linear is True:
        for name, param in model.named_parameters():
            param.requires_grad = False

    tte_model = TravelTimeEvaluator(model, args)
    tte_model = tte_model.to(device=args.device, non_blocking=True)

    optim = get_optimizer(tte_model.parameters(), args)
    sched = get_scheduler(optimizer=optim, args=args)

    writer = SummaryWriter(tsbd_dir)
    best_epoch = -1
    best_loss = 1e10
    patience = 0
    for ep in range(args.n_epochs):
        pbar = tqdm(train_loader, ncols=100, desc='Train')
        model.train()
        sched.step(ep)
        train_epoch_loss = []
        train_preds_list = []
        train_labels_list = []
        for batch_data, batch_label in pbar:
            for k, v in batch_data.items():
                batch_data[k] = v.cuda(non_blocking=True)
            batch_label = batch_label.cuda(non_blocking=True)

            optim.zero_grad()
            preds = tte_model(batch_data)
            loss = F.mse_loss(preds, batch_label)
            loss.backward()
            optim.step()

            with torch.no_grad():
                train_epoch_loss.append(loss.item())
                train_preds_list.append(preds.detach().cpu().numpy())
                train_labels_list.append(batch_label.detach().cpu().numpy())

            pbar.set_description(f"Epoch: {ep} | Train loss: {loss:.8f}")

        train_epoch_loss = np.mean(np.array(train_epoch_loss))
        train_results = travel_time_evaluation(train_preds_list, train_labels_list)

        eval_epoch_loss, eval_results = eval_fn(ep, tte_model, eval_loader)

        epoch_lr = optim.state_dict()['param_groups'][0]['lr']
        logger.info(f"Epoch: {ep:<2} | lr: {epoch_lr:>11.8f}")
        logger.info(f"Train loss: {train_epoch_loss:>12.8f} | train results: {train_results}")
        logger.info(f"Eval  loss: { eval_epoch_loss:>12.8f} | eval  results: { eval_results}")

        writer.add_scalar('lr', epoch_lr, ep)
        writer.add_scalar('train loss', train_epoch_loss, ep)
        writer.add_scalar('eval loss', eval_epoch_loss, ep)

        save_path = os.path.join(model_save_dir, f"model_{ep}.pt")
        save_checkpoint(save_path, tte_model, optim, sched)
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
    tte_model.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True)['model'])
    tte_model.to(device=args.device, non_blocking=True)

    test_results = test_fn(tte_model, test_loader)
    logger.info(f"Test results: {test_results}")
