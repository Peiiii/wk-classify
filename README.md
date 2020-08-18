# wk-classify
A package of tools for building deep-learning classification programs. Easy to use, light and powerful.

# Install
```shell script
pip3 install wk-classify
```

# Usage

### quick experience
```python
from wcf.packages.resnet.training import train, BaseConfig
class Config(BaseConfig):
    TRAIN_DIR = 'path for train set'
    VAL_DIR = 'path for val set'
cfg=Config()
train(cfg)
```
### a real example
```python
from wcf.packages.resnet.training import train, BaseConfig
from torchvision import transforms
class Config(BaseConfig):
    GEN_CLASSES_FILE = True
    USE_tqdm_TRAIN = False # use tqdm to format output
    INPUT_SIZE = (252,196)
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    BALANCE_CLASSES = True
    VAL_INTERVAL = 0.2 # val time insterval: 0.2 epoch (0.2* num_batches_per_epoch)
    WEIGHTS_SAVE_INTERVAL = 0.2 #  the same as above
    TRAIN_DIR = '<your train path>'
    VAL_DIR = '<your val path>'
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5),
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
cfg=Config()
train(cfg)
```

### all options
check out the `BaseConfig` class for all options

### how to predict?
check out `demo_predict.py`


