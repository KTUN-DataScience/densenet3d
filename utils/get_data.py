from dataset.jester import Jester
from config import Config

def get_training_set(spatial_transform, temporal_transform,
                     target_transform):
    return Jester(
        Config.dataset_path,
        Config.annotation_path,
        'training',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=Config.sample_duration)


def get_validation_set(spatial_transform, temporal_transform,
                       target_transform):
    return Jester(
            Config.dataset_path,
            Config.annotation_path,
            'validation',
            Config.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=Config.sample_duration)

def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    return Jester(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)