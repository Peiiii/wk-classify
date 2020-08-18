from wcf.packages.resnet.predict import Predictor,BasePredictConfig
import os,shutil,glob
from PIL import Image


def demo():
    predictor = Predictor(
        BasePredictConfig(WEIGHTS_PATH='weights/model_best.pkl',
        CLASSES_PATH='classes.txt')
    )
    data_dir='/home/ars/sda5/data/chaoyuan/datasets/classify_datasets/公章/val/0'
    out_dir=data_dir+'_result'
    import wk
    wk.remake(out_dir)
    fs=glob.glob(data_dir+'/*.jpg')
    # fs=glob.glob(data_dir+'/*.png')
    for i,f in enumerate(fs):
        img=Image.open(f)
        cls,prob=predictor.predict(img)
        cls_dir=out_dir+'/'+cls
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)
        f2=cls_dir+'/'+'%s-%s.jpg'%(i,cls)
        shutil.copy(f,f2)
        print(i,f,f2)

if __name__ == '__main__':
    demo()
