import torch
from torch import nn

def set_criterion():
    """Set criterion
    Description:
        - Set criterion based on whether cuda is available or not. 
    TODO: allow use of different criterion
    """
    criterion = nn.CrossEntropyLoss()
    return criterion.cuda() if torch.cuda.is_available() else criterion

def init_model(model):
    """Set model training devices. 
    Arguments: 
        model (class): This class must have a getModel function impletemented
    Returns:
        return pytorch model
    """
    model = model.getModel().cuda()
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
