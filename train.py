import warnings
import os
import argparse

import yaml
from loguru import logger

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import Config, get_optimizer, get_scheduler, init_seeds, save_checkpoint, load_checkpoint
from datasets import get_dataloader
from model.network import Net


@torch.no_grad()
def eval(args, ep, model, data_loader, writer=None):
    model.eval()
    batch_steps = len(data_loader.dataset) // args.batch_size
    epoch_loss = []
    epoch_spa_loss = []
    epoch_time_loss = []
    pbar = tqdm(data_loader, ncols=100)
    for iter_idx, batch_data in enumerate(pbar):
        for k, v in batch_data.items():
            batch_data[k] = v.cuda(non_blocking=True)
        spa_loss, time_loss = model.forward_loss(batch_data)
        loss = spa_loss + time_loss

        pbar.set_description(f"Epoch: {ep} | loss: {loss:.8f}")

        epoch_loss.append(loss.item())
        epoch_spa_loss.append(spa_loss.item())
        epoch_time_loss.append(time_loss.item())
        writer.add_scalar('eval_step_loss', loss.item(), batch_steps * ep + iter_idx)
        writer.add_scalar('eval_spa_loss', spa_loss.item(), batch_steps * ep + iter_idx)
        writer.add_scalar('eval_time_loss', time_loss.item(), batch_steps * ep + iter_idx)

    epoch_loss = np.mean(np.array(epoch_loss))
    epoch_spa_loss = np.mean(np.array(epoch_spa_loss))
    epoch_time_loss = np.mean(np.array(epoch_time_loss))
    return epoch_loss, epoch_spa_loss, epoch_time_loss


def train(parser_args):
    with open('config/' + parser_args.config + '.yaml', 'r') as f:
        args = yaml.full_load(f)
    args['device'] = parser_args.device
    args['exp_id'] = parser_args.exp_id
    args = Config(args)

    model_dir = os.path.join(args.save_dir, args.exp_id, "ckpts")
    tsbd_dir = os.path.join(args.save_dir, args.exp_id)
    log_dir = os.path.join(args.save_dir, args.exp_id)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tsbd_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tsbd_dir)
    logger.add(sink=f'{log_dir}/pretraining.log', mode='w',
               format="[{time:YYYY-M-D HH:mm:ss}] [{module}:{line} {function}()] -> {message}")
    logger.info(args)

    model = Net(args=args).to(device=args.device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Params: {num_params}")

    train_loader, eval_loader = get_dataloader(args)

    optim = get_optimizer(model.parameters(), args)
    sched = get_scheduler(optimizer=optim, args=args)

    if args.start_epoch != 0:
        load_path = os.path.join(model_dir, f"model_{args.start_epoch - 1}.pt")
        print("loading model at", load_path)
        load_checkpoint(load_path, model, optim, sched)
        model.to(device=args.device)

    batch_steps = len(train_loader.dataset) // args.batch_size

    for ep in range(args.start_epoch, args.n_epochs):
        pbar = tqdm(train_loader, ncols=100)
        epoch_loss = []
        epoch_spa_loss = []
        epoch_time_loss = []
        model.train()
        sched.step(ep)
        for iter_idx, batch_data in enumerate(pbar):
            for k, v in batch_data.items():
                batch_data[k] = v.cuda(non_blocking=True)
            optim.zero_grad()

            spa_loss, time_loss = model.forward_loss(batch_data)

            loss = spa_loss + time_loss

            loss.backward()
            optim.step()
            pbar.set_description(f"Epoch: {ep} | loss: {loss:.8f}")
            epoch_loss.append(loss.item())
            epoch_spa_loss.append(spa_loss.item())
            epoch_time_loss.append(time_loss.item())

            writer.add_scalar('step_loss', loss.item(), batch_steps * ep + iter_idx)
            writer.add_scalar('spa_loss', spa_loss.item(), batch_steps * ep + iter_idx)
            writer.add_scalar('time_loss', time_loss.item(), batch_steps * ep + iter_idx)

            del batch_data
            torch.cuda.empty_cache()

        eval_loss, eval_spa_loss, eval_time_loss = eval(args, ep, model, eval_loader, writer)

        epoch_lr = optim.state_dict()['param_groups'][0]['lr']
        epoch_loss = np.mean(np.array(epoch_loss))
        epoch_spa_loss = np.mean(np.array(epoch_spa_loss))
        epoch_time_loss = np.mean(np.array(epoch_time_loss))

        logger.info(f"Train epoch: {ep:<2} | lr: {epoch_lr:.8f}")
        logger.info(f"Train loss: {epoch_loss:>11.8f} | spa_loss: {epoch_spa_loss:>11.8f} | time_loss: {epoch_time_loss:>11.8f}")
        logger.info(f"Eval  loss: {eval_loss:>11.8f} | spa_loss: {eval_spa_loss:>11.8f} | time_loss: {eval_time_loss:>11.8f}")

        writer.add_scalar('lr', epoch_lr, ep)
        writer.add_scalar('loss', epoch_loss, ep)
        writer.add_scalar('eval_loss', eval_loss, ep)
        writer.add_scalar('eval_spa_loss', eval_spa_loss, ep)
        writer.add_scalar('eval_time_loss', eval_time_loss, ep)

        if (args.save_per != 0 and ep % args.save_per == 0) or ep == args.n_epochs - 1:
            save_path = os.path.join(model_dir, f"model_{ep}.pt")
            save_checkpoint(save_path, model, optim, sched)
            print('saved model at', save_path)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='porto')
    parser.add_argument("--exp_id", type=str, default='enlayer242_cross_bz256_epoch30_dim128_attn_1e-4')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    init_seeds()
    torch.cuda.set_device(args.device)
    train(args)
