import warnings
import argparse
import yaml

import torch
from utils import Config, init_seeds


def read_config(parser_args):
    with open('config/' + parser_args.config + '.yaml', 'r') as f:
        args = yaml.full_load(f)
    for k, v in parser_args.__dict__.items():
        args[k] = v
    args = Config(args)

    if parser_args.transfer is True:
        if parser_args.config == parser_args.transfer_config:
            raise ValueError('The source and target config files are the same.')

        with open('config/' + parser_args.transfer_config + '.yaml', 'r') as f:
            transfer_args = yaml.full_load(f)
            transfer_args = Config(transfer_args)
            args.root = transfer_args.root
            args.dataset_name = transfer_args.dataset_name
            args.traj_file = transfer_args.traj_file
            args.task = args.task + '_' + args.config + '2' + args.transfer_config
            args.mbr = transfer_args.mbr
            args.center = transfer_args.center
            args.max_patch_len_s3 = transfer_args.max_patch_len_s3
            args.max_patch_len_s2 = transfer_args.max_patch_len_s2
    return args


def set_classification_config(parser_args):
    args = read_config(parser_args)

    args.lr = 1e-4

    args.optim = 'Adam'

    args.scheduler = 'cos'

    args.n_epochs = 30
    args.warmup_epoch = 10
    args.warmup_lr_init = 1.0e-6
    args.lr_min = 1.0e-6

    if args.dataset_name.startswith('porto'):
        args.n_classes = 3
    elif args.dataset_name.startswith('chengdu'):
        args.n_classes = 2
    else:
        raise ValueError('Invalid dataset name')

    args.patience = 7

    return args


def set_travel_time_config(parser_args):
    args = read_config(parser_args)

    args.lr = 1e-4

    args.optim = 'Adam'

    args.scheduler = 'cos'
    args.n_epochs = 30
    args.warmup_epoch = 10
    args.warmup_lr_init = 1.0e-6
    args.lr_min = 1.0e-6

    args.patience = 7

    return args


def set_similarity_config(parser_args):
    return read_config(parser_args)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer", type=bool, default=False)
    parser.add_argument("--transfer_config", type=str, default='chengdu')

    parser.add_argument("--config", type=str, default='chengdu')
    parser.add_argument("--exp_id", type=str, default='enlayer242_cross_bz256_epoch30_dim128_attn_1e-4')
    parser.add_argument("--task", type=str, default='similarity')

    parser.add_argument("--model_id", type=int, default=29)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    init_seeds()
    torch.cuda.set_device(args.device)

    if args.task == 'similarity':
        from downstream.similarity import train
        args = set_similarity_config(args)
        train(args)
    elif args.task == 'classification':
        from downstream.classification import train
        args = set_classification_config(args)
        train(args)
    elif args.task == 'travel_time':
        from downstream.travel_time import train
        args = set_travel_time_config(args)
        train(args)
    else:
        raise ValueError('Invalid task')
