import numpy as np
from torchvision import transforms
from wpcv.utils.data_aug import img_aug, random_float_generator
from wpcv.utils.ops import pil_ops
from wcf.packages.resnet.training import train, BaseConfig

class Config(BaseConfig):
    GEN_CLASSES_FILE = True
    USE_tqdm_TRAIN = False
    INPUT_SIZE = (252,196)
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    BALANCE_CLASSES = True
    VAL_INTERVAL = 0.2
    WEIGHTS_SAVE_INTERVAL = 0.2
    TRAIN_DIR = 'datasets/classify_datasets/公章/train'
    VAL_DIR = 'datasets/classify_datasets/公章/val'
    train_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5),
        img_aug.RandomApply(pil_ops.shear_xy,
                            random_params=dict(degree1=random_float_generator(5), degree2=random_float_generator(5))),
        img_aug.RandomApply(pil_ops.rotate, random_params=dict(degree=random_float_generator(5))),
        img_aug.RandomApply(pil_ops.translate,
                            random_params=dict(offset=random_float_generator(10, shape=(2,), dtype=np.int))),
        img_aug.RandomApply(pil_ops.blur, p=0.3),
        img_aug.RandomApply(pil_ops.sp_noise),
        img_aug.RandomApply(pil_ops.edge_enhance, p=0.3),
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if __name__ == '__main__':
    cfg = Config()
    train(cfg)
