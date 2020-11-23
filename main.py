import os 
import torch
from datetime import datetime
from torch import optim
from torch.optim import lr_scheduler
from config import Config
from model import densenet
from utils.get_data import *
from utils.train_utils import *
from utils.spatial_transforms import *
from utils.temporal_transforms import *
from utils.train import train_epoch, val_epoch, test, evaluate_model
from utils.target_transforms import ClassLabel, VideoID
from utils.target_transforms import Compose as TargetCompose

if __name__ == "__main__":

    # start time
    import pdb; pdb.set_trace()
    start =  datetime.now()

    print(torch.cuda.is_available())

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    print(f'Graphic Cart Used for the experiment: {device}')

    model = init_model(densenet)

    criterion = set_criterion()
    
    mean = get_mean(Config.norm_value, dataset = Config.mean_dataset) 

    std = get_std(Config.norm_value)

    store_name = '_'.join([Config.dataset, Config.model, str(Config.width_mult) + 'x',
                               Config.modality, str(Config.sample_duration)])

    norm_method = set_norm_method(mean, std)

    # set scaling values
    scales = [Config.initial_scale]
    for i in range(1, Config.n_scales):
        scales.append(scales[-1] * Config.scale_step)

    if Config.train:
        crop_method = set_crop_method(scales)

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

        # TODO: create optmizer function to set on the fly a selected optmizer

        optimizer = optim.SGD(
            model.parameters(),
            lr=Config.learning_rate,
            momentum=Config.momentum,
            dampening=dampening,
            weight_decay=Config.weight_decay,
            nesterov=Config.nesterov)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
            patience=Config.lr_patience)

    # If set validation set. Apply spatial transformations. 

    if Config.validation:
        spatial_transform = Compose([
            Scale(Config.sample_size),
            CenterCrop(Config.sample_size),
            ToTensor(Config.norm_value), norm_method
        ])
        #temporal_transform = LoopPadding(opt.sample_duration)
        temporal_transform = TemporalCenterCrop(Config.sample_duration, Config.downsample)
        target_transform = ClassLabel()
        validation_data = get_validation_set(spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(Config.result_path, 'val.log'), ['epoch', 'loss', 'prec1', 'prec5'])

    best_prec1 = 0
    if Config.resume_path:
        print('loading checkpoint {}'.format(Config.resume_path))
        checkpoint = torch.load(Config.resume_path)
        assert Config.arch == checkpoint['arch']
        best_prec1 = checkpoint['best_prec1']
        Config.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    
    for i in range(Config.begin_epoch, Config.n_epochs + 1):
        if Config.train:
            adjust_learning_rate(optimizer, i)
            
            train_epoch(i, train_loader, model, criterion, optimizer,
                            train_logger, train_batch_logger)
            state = {
                'epoch': i,
                'arch': Config.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            save_checkpoint(state, False, store_name)

        if  Config.validation:
            validation_loss, prec1 = val_epoch(i, val_loader, model, criterion,
                val_logger)
                 
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            state = {
                'epoch': i,
                'arch': Config.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            save_checkpoint(state, is_best, store_name)
    
    if Config.test:

        evaluate_model(densenet)

    #     spatial_transform = Compose([
    #         Scale(int(Config.sample_size / Config.scale_in_test)),
    #         CornerCrop(Config.sample_size, Config.crop_position_in_test),
    #         ToTensor(Config.norm_value), norm_method
    #     ])
    #     # temporal_transform = LoopPadding(opt.sample_duration, opt.downsample)
    #     temporal_transform = TemporalRandomCrop(Config.sample_duration, Config.downsample)
    #     target_transform = VideoID()

    #     test_data = get_test_set(spatial_transform, temporal_transform, target_transform)
    #     test_loader = torch.utils.data.DataLoader(
    #         test_data,
    #         batch_size=Config.batch_size,
    #         shuffle=False,
    #         num_workers=Config.n_threads,
    #         pin_memory=True)
    #     test(test_loader, model, test_data.class_names)

    # time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
        
    