import pickle
import random
import numpy as np
import torch
from timm.scheduler import CosineLRScheduler, StepLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR


def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def init_seeds(RANDOM_SEED=42):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config(object):
    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])

    def __str__(self):
        dic = self.__dict__.copy()
        lst = list(filter(
            lambda p: (not p[0].startswith('__')) and not isinstance(p[1], classmethod),
            dic.items()
        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, save_path=None, dp_flag=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.dp_flag = dp_flag
        self.best_epoch = -1

    def __call__(self, val_loss, model, classifier=None, time_predictor=None, decoder=None, epoch=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, time_predictor, decoder, dp_flag=self.dp_flag)
            if epoch is not None:
                self.best_epoch = epoch
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, time_predictor, decoder, dp_flag=self.dp_flag)
            if epoch is not None:
                self.best_epoch = epoch
            self.counter = 0

    def save_checkpoint(self, val_loss, model, classifier=None, time_predictor=None, decoder=None, dp_flag=False):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        classifier_state_dict = None

        if dp_flag:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        if classifier is not None:
            classifier_state_dict = classifier.state_dict()

        if self.save_path is not None:
            torch.save({
                'model_state_dict': model_state_dict,
                'classifier_state_dict': classifier_state_dict,
            }, self.save_path)
        else:
            print("no path assigned")

        self.val_loss_min = val_loss


def save_checkpoint(save_path, model, optim, sched):
    checkpoint = {
        'model': model.state_dict(),
        # 'optim': optim.state_dict(),
        # 'sched': sched.state_dict(),
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(load_path, model, optim, sched):
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])
    sched.load_state_dict(checkpoint['sched'])


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmt_str = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmt_str.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_optimizer(parameters, args):
    if not hasattr(args, 'optim'):
        return torch.optim.Adam(params=parameters, lr=args.lr, **args.optim_args)
    elif args.optim == 'AdamW':
        return torch.optim.AdamW(params=parameters, lr=args.lr, **args.optim_args)
    elif args.optim == 'Adam':
        return torch.optim.Adam(params=parameters, lr=args.lr)
    else:
        raise NotImplementedError('No Optimizer!')


def get_scheduler(optimizer, args):
    if not hasattr(args, 'scheduler'):
        return CosineLRScheduler(optimizer=optimizer,
                                 t_initial=args.n_epochs,
                                 warmup_t=args.warmup_epoch,
                                 warmup_lr_init=args.warmup_lr_init,
                                 lr_min=args.lr_min
                                 )
    elif args.scheduler == 'cos':
        return CosineLRScheduler(optimizer=optimizer,
                                 t_initial=args.n_epochs,
                                 warmup_t=args.warmup_epoch,
                                 warmup_lr_init=args.warmup_lr_init,
                                 lr_min=args.lr_min
                                 )
    elif args.scheduler == 'step':
        return StepLRScheduler(optimizer=optimizer,
                               decay_rate=args.decay_rate,
                               decay_t=args.decay_t,
                               warmup_t=args.warmup_epoch,
                               warmup_lr_init=args.warmup_lr_init
                               )
    elif args.scheduler == 'cos_ann':
        return CosineAnnealingLR(optimizer=optimizer, T_max=args.n_epochs, eta_min=args.lr_min)
    else:
        raise NotImplementedError('No Scheduler!')
