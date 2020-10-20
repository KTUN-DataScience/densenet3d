class Config:
    """Model training Configuration class
    Attributes:
        learning_rate (float) - how many filters to add each layer (k in paper)
        epoch (int) - how many layers in each pooling block
    Usage:
        from config import Config
        Config.learning_rate
    """
    
    learning_rate = 0.01
    epoch = 10
    optimiser = 'SDG'
    cuda: True