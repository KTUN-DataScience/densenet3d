import torch
from torch import nn
import csv
import shutil
import numpy as np
from config import Config

def set_criterion():
    """Set criterion
    Description:
        - Set criterion based on whether cuda is available or not. 
    TODO: allow use of different criterion
    """
    criterion = nn.CrossEntropyLoss()
    return criterion.cuda()

def init_model(model):
    """Set model training devices. 
    Arguments: 
        model (class): This class must have a getModel function impletemented
    Returns:
        return pytorch model
    """
    model = model.getModel(
        num_classes=Config.n_classes
    ).cuda()
    model = nn.DataParallel(model, device_ids=None)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model


def get_mean(norm_value=255, dataset='activitynet'):
    assert dataset in ['activitynet', 'kinetics']

    if dataset == 'activitynet':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [
            110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value
        ]


def get_std(norm_value=255):
    # Kinetics (10 videos for each class)
    return [
        38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value
    ]

class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, store_name):
    torch.save(state, '%s/%s_checkpoint.pth' % (Config.result_path, store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (Config.result_path, store_name),'%s/%s_best.pth' % (Config.result_path, Config.store_name))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = Config.learning_rate * (0.1 ** (sum(epoch >= np.array(Config.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        #param_group['lr'] = opt.learning_rate
