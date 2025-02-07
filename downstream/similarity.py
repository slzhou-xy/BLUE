import os
from loguru import logger

import numpy as np
import torch
from tqdm import tqdm

from model.network import Net
from datasets import get_dataloader

from downstream.downstream_utils import mr, hit_ratio


@torch.no_grad()
def inference_fn(model, data_loader, desc):
    model.eval()
    pbar = tqdm(data_loader, ncols=100, desc=desc)
    embeddings = []
    for batch_data in pbar:
        for k, v in batch_data.items():
            batch_data[k] = v.cuda(non_blocking=True)
        emb = model.forward_encoder(batch_data)
        embeddings.append(emb.detach().cpu().numpy())
    return np.vstack(embeddings)


def train(args):
    pretrained_model_dir = os.path.join(args.save_dir, args.exp_id, 'ckpts')
    log_dir = os.path.join(args.save_dir, args.exp_id, 'task', args.task)

    os.makedirs(log_dir, exist_ok=True)
    logger.add(sink=f'{log_dir}/{args.task}.log', mode='w',
               format="[<red>{time:YYYY-M-D HH:mm:ss}</red>] [{module}:{line} {function}()] -> {message}")
    logger.info(args)

    model = Net(args=args)
    load_path = os.path.join(pretrained_model_dir, f"model_{args.model_id}.pt")
    print("loading model at", load_path)
    model.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True)['model'], strict=False)
    model.to(device=args.device, non_blocking=True)

    database_loader, label_loader, query_loader = get_dataloader(args, args.task)
    database_emb = inference_fn(model=model, data_loader=database_loader, desc='Database')
    label_emb = inference_fn(model=model, data_loader=label_loader, desc='Label   ')
    query_emb = inference_fn(model=model, data_loader=query_loader, desc='Query   ')

    full_databse_emb = np.vstack([label_emb, database_emb])
    dists = query_emb @ full_databse_emb.T
    scores = np.argsort(dists, axis=-1)[:, ::-1][:, :10]
    truth = list(range(database_emb.shape[0]))

    mr_res = mr(dists)
    hr_res = hit_ratio(truth, scores, [1, 5, 10])

    logger.info(f'MR: {mr_res}')
    logger.info(f'HR: {hr_res}')
