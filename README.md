# Densenet3d

## Introduction
Densenet3D implementation using PyTorch

## Getting started
---

### Requirements
- [python 3.6 >=][pythorn] 
- CUDA [10.1][cuda10.1] or [10.2][cuda10.2]
- [pytorch 1.6.0][pytorch]
- openCV
- scipy
- Numpy

### Dataset
Download 20BN-Jester Dataset from [here](https://20bn.com/datasets/download)

Use the following credentials to login if you do not want to create your own.
```text
email: pikemsonda@yahoo.com 
password: jesterdataset
```

After you log in chose the following dataset and download to a folder. 

[dataset]: https://github.com/KTUN-DataScience/densenet3d/raw/main/docs/images/dataset.jpg

After download extract:
```console
cd download_folder/

cat 20bn-jester-v1-??|gzip -dc|tar xf - 
```
For the second command use a unix terminal like git bash. After extraction add the folder to the project or place the folder in  place you can easily access.

###  Install Dependencies

Install dependencies using requirement.txt with the following command. 
```console
pip install -r requirements.txt
```
### Create frames

```console
python utils/n_frames_jester.py dataset folder
```

### Configuration 
`config.py` contains all the parameters to train **Densenet3D** model
| Attribute        | Default           | Description  |
| ------------- |:-------------:| :-----|
| `arch`     | Densenet | Name of the architecture of the model |
| `learning_rate`      | 0.01      | Set learning rate for the model |

**Example**:
```python
class Config:
    """
    Model training Configuration class
    Attributes:
        arch: Name of architecture being used
        learning_rate (float) - how many filters to add each layer (k in paper)
        n_epoch (int) - how many layers in each pooling block
        lr_patience:
    """
    arch = 'Densenet'

    learning_rate = 0.01

    lr_patience = 10

    momentum = 0.9

    begin_epoch = 1

    n_epochs = 2

    n_classes = 5
    ...
``` 

### Run
```console
python main.py
```
## References

1. [Resource Efficient 3D Convolutional Neural Networks][1]
2. [Densely Connected Convolutional Networks][2]

## Acknoweledgements
This project was setup with inspiration from the [Kensho Hara](https://github.com/kenshohara/3D-ResNets-PyTorch) and [Okan Köpüklü](https://github.com/okankop/Efficient-3DCNNs)


[cuda10.1]: https://developer.nvidia.com/cuda-10.1-download-archive-base

[cuda10.2]: https://developer.nvidia.com/cuda-10.2-download-archive

[pytorch]: https://pytorch.org/get-started/locally/

[pythorn]: https://www.python.org/downloads/

[1]: https://arxiv.org/pdf/1904.02422.pdf
[2]: https://arxiv.org/abs/1608.06993