import torch
from config import Config
from model import densenet

if __name__ == "__main__":
    print(densent.model())
    device = torch.device('cuda' if Config.cuda else 'cpu')
    print(device)
    print(torch.cuda.device_count())
    pass