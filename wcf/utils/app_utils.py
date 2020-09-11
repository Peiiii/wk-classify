from wcf import train, TrainValConfigBase, val, t, EasyTransform, models_names
from wk import PointDict
from vtgui.app import make_app, SelectDir, SelectFile, VirtualField
from wcf import models_names
import sys


def make_trainval_config(
        cfg, data_cfg
):
    cfg=PointDict(**cfg)
    data_cfg=PointDict(**data_cfg)
    class Config(TrainValConfigBase):
        MODEL_TYPE = cfg.MODEL_TYPE
        GEN_CLASSES_FILE = cfg.GEN_CLASSES_FILE
        USE_tqdm_TRAIN = cfg.USE_tqdm_TRAIN
        INPUT_SIZE = (cfg.INPUT_W, cfg.INPUT_H)
        BATCH_SIZE = cfg.BATCH_SIZE
        NUM_EPOCHS = cfg.NUM_EPOCHS
        BALANCE_CLASSES = cfg.BALANCE_CLASSES
        VAL_INTERVAL = cfg.VAL_INTERVAL
        WEIGHTS_SAVE_INTERVAL = cfg.WEIGHTS_SAVE_INTERVAL
        WEIGHTS_INIT = cfg.WEIGHTS_INIT
        TRAIN_DIR = cfg.TRAIN_DIR
        VAL_DIR = cfg.VAL_DIR
        INPUT_W = cfg.INPUT_W
        INPUT_H = cfg.INPUT_H
        val_transform = EasyTransform([
            t.Resize(INPUT_SIZE[::-1]),
            t.SaveToDir(cfg.VISUALIZE_RESULT_DIR),
            t.ToTensor(),
        ])
        train_transform = EasyTransform(list(filter(lambda x:x is not None,[
            t.ColorJitter(brightness=data_cfg.BRIGHTNESS, contrast=data_cfg.CONTRAST, saturation=data_cfg.SATURATION, hue=data_cfg.HUE),
            t.RandomHorizontalFlip() if data_cfg.RandomHorizontalFlip else None,
            t.RandomVerticalFlip() if data_cfg.RandomVerticalFlip else None,
            t.RandomRotate(data_cfg.RandomRotate) if data_cfg.RandomRotate else None ,
            t.RandomShear(data_cfg.RandomShear,data_cfg.RandomShear) if data_cfg.RandomShear else None,
            t.RandomTranslate(data_cfg.RandomTranslate) if data_cfg.RandomTranslate else None,
            t.RandomBlur(p=data_cfg.RandomBlur, radius=1) if data_cfg.RandomBlur else None,
            t.RandomSPNoise(p=data_cfg.RandomSPNoise) if data_cfg.RandomSPNoise else None,
            *val_transform,
        ])))
    return Config




models = [
    models_names.resnet10,
    models_names.resnet18,
    models_names.resnet50,
    models_names.shufflenet_v2_x0_5,
    models_names.shufflenet_v2_x1_0,
]

def get_base_config():
    base_config = dict(
        MODEL_TYPE=VirtualField(title='模型', description='选择模型', default='resnet18', options=models),
        TRAIN_DIR=SelectDir('/home/ars/sda5/data/projects/烟分类/data/烟分类-train'),
        VAL_DIR=SelectDir('/home/ars/sda5/data/projects/烟分类/data/烟分类-val'),
        GEN_CLASSES_FILE=VirtualField(default=True, title='生成类别文件'),
        USE_tqdm_TRAIN=True,
        BATCH_SIZE=64,
        NUM_EPOCHS=200,
        BALANCE_CLASSES=True,
        VAL_INTERVAL=1,
        WEIGHTS_SAVE_INTERVAL=1,
        WEIGHTS_INIT='weights/training/model_best.pkl',
        INPUT_W=224,
        INPUT_H=224,
        VISUALIZE_RESULT_DIR='data/visualize',
    )
    return base_config

def get_data_config():
    data_config = dict(
        BRIGHTNESS=0.1,
        CONTRAST=0.05,
        SATURATION=0.05,
        HUE=0.05,
        RandomHorizontalFlip=False,
        RandomVerticalFlip=False,
        RandomRotate=0,
        RandomShear=0,
        RandomTranslate=0,
        RandomBlur=0.3,
        RandomSPNoise=0.3,
    )
    return data_config


def training_callback(base_cfg, data_config):
    Config= make_trainval_config(base_cfg,data_config)
    cfg=Config()
    train(cfg)

def make_training_app():
    base_config=get_base_config()
    data_config=get_data_config()
    app = make_app(function=training_callback, args=(base_config, data_config), columns=[4, 3])
    return app


