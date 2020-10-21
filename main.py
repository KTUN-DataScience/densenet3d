import os 
import torch
from torch import optim
from torch.optim import lr_scheduler
from config import Config
from model import densenet
from utils.get_data import *
from utils.train_utils import *
from utils.spatial_transforms import *
from utils.temporal_transforms import *
from utils.target_transforms import ClassLabel, VideoID
from utils.target_transforms import Compose as TargetCompose

if __name__ == "__main__":
    torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = init_model(densenet)

    criterion = set_criterion()
    
    mean = get_mean(Config.norm_value, dataset = Config.mean_dataset)

    std = get_std(Config.norm_value)

    store_name = '_'.join([Config.dataset, Config.model, str(Config.width_mult) + 'x',
                               Config.modality, str(Config.sample_duration)])

    norm_method = Normalize(mean, std)

    crop_method = MultiScaleCornerCrop(Config.scales, Config.sample_size)

    spatial_transform = Compose([
        RandomHorizontalFlip(),
        #RandomRotate(),
        #RandomResize(),
        crop_method,
        #MultiplyValues(),
        #Dropout(),
        #SaltImage(),
        #Gaussian_blur(),
        #SpatialElasticDisplacement(),
        ToTensor(Config.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(Config.sample_duration, Config.downsample)
    target_transform = ClassLabel()
    training_data = get_training_set(spatial_transform,temporal_transform,target_transform)

    train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.n_threads,
            pin_memory=True)

    train_logger = Logger(
            os.path.join(Config.result_path, 'train.log'),
            ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
            
    train_batch_logger = Logger(
        os.path.join(Config.result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])

    dampening = Config.dampening
    if Config.nesterov:
        dampening = 0
    else:
        dampening = Config.dampening

    optimizer = optim.SGD(
        model.paramters(),
        lr=Config.learning_rate,
        momentum=Config.momentum,
        dampening=dampening,
        weight_decay=Config.weight_decay,
        nesterov=Config.nesterov)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
        patience=Config.lr_patience)