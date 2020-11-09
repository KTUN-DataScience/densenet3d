class Config:
    """
    Model training Configuration class
    Attributes:
        arch: Name of architecture being used
        learning_rate (float) - how many filters to add each layer (k in paper)
        n_epoch (int) - how many layers in each pooling block
        lr_patience:
        momentum:
        begin_epoch:
        n_epochs:
        lr_steps:
        weight_decay:
        nesterov:
        dataset:
        model:
        width_mult:
        batch_size:
        n_threads:
        dampening:
        modality:
        sample_duration:
        sample_size:
        n_val_samples:
        downsample:
        optimiser:
        cuda:
        dataset_path:
        annotation_path:
        result_path:
        n_classes:
        initial_scale:
        n_scales:
        scale_step:
        norm_value:
        mean_dataset:
        mean_norm:
        std_norm: 
    Usage:
        from config import Config
        Config.learning_rate
    """

    arch = 'Densenet'

    learning_rate = 0.01

    lr_patience = 10

    momentum = 0.9

    begin_epoch = 1

    n_epochs = 2

    n_classes = 5

    lr_steps = [40, 55, 65, 70, 200, 250]
    weight_decay = 1e-3
    nesterov = True
    dataset = 'jester'
    model = 'densenet3d'
    width_mult = 1.0
    batch_size = 16
    n_threads = 4
    dampening = 0.9
    modality = 'RGB'
    sample_duration = 16
    sample_size = 112
    n_val_samples = 3
    downsample = 1
    optimiser = 'SDG'
    cuda = True
    dataset_path = 'dataset/jester'
    annotation_path = 'dataset/annotation/jester.json'
    result_path = '\\results'
    initial_scale = 1
    n_scales = 5
    scale_step = 0.84089641525
    norm_value = 255
    mean_dataset = 'kinetics'
    dataset = ''
    store_name = ''
    train_crop = ''
    mean_norm = False,
    std_norm = False
    labels_to_use = {
        'labels': [
            # "Doing_other_things",
            # "Drumming_Fingers",
            # "No_gesture",
            # "Pulling_Hand_In",
            # "Pulling_Two_Fingers_In",
            "Pushing_Hand_Away",
            "Pushing_Two_Fingers_Away",
            "Rolling_Hand_Backward",
            "Rolling_Hand_Forward",
            "Shaking_Hand",
            # "Sliding_Two_Fingers_Down",
            # "Sliding_Two_Fingers_Left",
            # "Sliding_Two_Fingers_Right",
            # "Sliding_Two_Fingers_Up",
            # "Stop_Sign",
            # "Swiping_Down",
            # "Swiping_Left",
            # "Swiping_Right",
            # "Swiping_Up",
            # "Thumb_Down",
            # "Thumb_Up",
            # "Turning_Hand_Clockwise",
            # "Turning_Hand_Counterclockwise",
            # "Zooming_In_With_Full_Hand",
            # "Zooming_In_With_Two_Fingers",
            # "Zooming_Out_With_Full_Hand",
            # "Zooming_Out_With_Two_Fingers"
        ]
    }
