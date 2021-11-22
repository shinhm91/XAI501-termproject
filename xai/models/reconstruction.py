#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:49:24 2021

@author: juliashin
"""


# -------------------------------------
# [Empty Module #1] Reconstruction based anomaly detection
# -------------------------------------
# 
# ------------------------------------------------------------
# 구현 가이드라인 
# ------------------------------------------------------------
# [1] train.imgs와 val.imgs =  정규화.
# [2] PCA 초기화 및 train imgs를 학습
# Baseline : PCA(n_components=n_components), random_state=777
# [3] val imgs를 transform.
# [4] [3]로 얻은 val_pca를 inverse_transform을 통해 복원. scaler의 
# inverse_transform을 진행 후, (num_data_imgs, img_width, img_height)로 reshape.
# [5] Reconstruction_error = Original imgs - Reconstruction imgs
# Original imgs : reshape (num_data_imgs, img_width, img_height)를 해야합니다.
 # ------------------------------------------------------------
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import umap # pip install umap-learn

# -------------------------------------
# Load preprocessed data
# -------------------------------------
  
#sc = StandardScaler()
#train.imgs = sc.fit_transform(train.imgs)
#val.imgs = sc.transform(val.imgs)


class Reconstruction():
    """
    Class for implementing reconstruction based anomaly detection
    
    여기 부분 한번 봐주세요
    """
    def __init__(self):
        self.learned = False
        self.optimal = None
        
    def fit(self, train.imgs):
        """
        여기 부분 한번 봐주세요
        """
        Reconstruction_imgs = sc.inverse_transform(val.imgs)
        Reconstruction_imgs = Reconstruction_imgs.reshape(len(Reconstruction_imgs),86,86)
        
        
        self.learned = True
        return self

    def predict(self, test.imgs):
        """
        여기 부분 한번 봐주세요
        """
        if not self.learned:
            raise NameError('Fit model first')

        ori = val.imgs.reshape(len(val.imgs), 86, 86)

        Reconstruction_error = ori - Reconstruction_imgs
        cls_score = Reconstruction_error.sum(axis=1).sum(axis=1)
        cls_score = sc.fit_transform(cls_score.reshape(-1, 1))

        y_pred = cls_score

        th = 0.5
        y_pred[cls_score < th] = -1
        y_pred[cls_score > th] = 1
        y_pred = y_pred.reshape(-1)
        
        return y_pred
    
    

#pca = PCA(n_components=n_components, random_state=777)
#pca.fit(train.imgs)
#val_pca = pca.transform(val.imgs)




