import os
import pickle
import pandas as pd
from easydict import EasyDict as edict
from glob import glob
import numpy as np
from PIL import Image


class MVTec_AD():
    def __init__(self, DB_PATH, OUT_PATH, size=(86,86), flatten=True):
        '''
        DB_PATH : str. e.g. {workspace}/dataset
        size : (int, int). default (86, 86)
        flatten : bool. [num_of_img_per_class, h*w]/[num_of_img_per_class, h, w]
        '''
        self.DB_PATH = DB_PATH
        self.out_path = OUT_PATH
        self.size, self.flatten = size, flatten

        self.class_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                            'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                            'tile', 'transistor', 'wood', 'zipper']


    def read_mvtec(self, cls="bottle", mode='train'):
        '''
        input
          cls : str. class name.
          mode : str. {train, test, val}
        output
          db_dict : dict. 
                  if val in mode
                   {'imgs':[num_of_img_per_class, h*w or h, w], 'labels':gt, 'class_name':class_name}
                  else
                   {'imgs':[num_of_img_per_class, h*w or h, w], 'class_name':class_name}
        '''

        # Load cache data.
        CACHE_PATH = self.out_path + '/cache'
        cache_ = os.path.join(CACHE_PATH, mode)
        if self.flatten: cache_+= 't_'
        cache_+=f'{str(self.size)}_{cls}.pkl'

        # if exist cache data return data
        if os.path.isfile(cache_):
            with open(cache_, 'rb') as f:
                data = pickle.load(f)
            return data
        # else load imgs and cache data
        else:
            if os.path.isdir(CACHE_PATH)!=True:
                os.mkdir(CACHE_PATH)

            # load imgs
            data = edict()  
            if mode == 'val':
                csv = pd.read_csv(os.path.join(self.DB_PATH, cls, 'val.csv'))
                data.imgs = [f'{self.DB_PATH}/{cls}/test/{csv.iloc[id, 0]:03d}.png' for id in csv.index]
                data.labels = [csv.iloc[id, 1] for id in csv.index]
            else:
                data.imgs = sorted(glob(os.path.join(self.DB_PATH, cls, mode, '*.png')))

            data.class_name = cls

            data = self.read_img(data)

            # cache data
            with open(cache_, "wb") as f:
                pickle.dump(data, f)

            return data
        
    def read_img(self, db_dict):
        im_path = db_dict.imgs
        len_imgs = len(im_path)

        db_dict.imgs = np.array([np.array(Image.open(im).convert("L").resize(self.size)) for im in im_path])
        if self.flatten:
            db_dict.imgs = db_dict.imgs.reshape(len_imgs,-1)
      
        return db_dict