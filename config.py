class Config:
    """Model training Configuration class
    Attributes:
        learning_rate (float) - how many filters to add each layer (k in paper)
        epoch (int) - how many layers in each pooling block
    Usage:
        from config import Config
        Config.learning_rate
    """
    
    # Set learning rate of model.
    learning_rate = 0.01
    lr_patience = 10
    momentum = 0.9
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
    epoch = 10
    optimiser = 'SDG'
    cuda = True
    dataset_path = ''
    annotation_path = ''
    result_path = ''
    number_of_classes: 27
    initial_scale = 1
    scales = 5
    scale_step = 0.84089641525
    norm_value = 255
    mean_dataset = 'kinetics'
    dataset = ''
    store_name = ''
    train_crop = ''
    mean_norm = False,
    std_norm = False
