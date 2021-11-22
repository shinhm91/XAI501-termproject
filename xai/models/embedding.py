#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:06:01 2021

@author: juliashin
"""
# -------------------------------------
# [Empty Module #2]
# -------------------------------------
# ------------------------------------------------------------
# 구현 가이드라인 
# ------------------------------------------------------------
# Empty Module
# [0] classification module과 pca 모델 초기화
#  사용된 PCA. e.g. PCA, RandomizePCA, KernelPCA...
# Baseline : sklearn.svm.OneClassSVM, kernel="rbf", gamma=0.001, nu=0.01
# Baseline : PCA(n_components=n_components), random_state=777
# RandomizePCA ~(svd_solver='randomized') 사용시, whiten=True로 사용.
# [1] train.imgs와 val.imgs 정규화.
# [2] PCA 초기화와 정규화된 train.imgs를 학습 후, 정규화된 train.imgs, val imgs를 transform.
# [3] OneClassSVM을 이용하여 차원 축소된 train 학습 후 val data 추론. 
# hint. OneClassSVM.predict, OneClassSVM.score_samples 사용.
# [4] test.imgs에 [1]~[3]과 동일한 과정을 적용. 단, 추론 시 score에 대해서만 계산

from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import umap # pip install umap-learn

# -------------------------------------
# Load preprocessed data
# -------------------------------------

class oneclasssvm():
    """
    """
    
    def __init__(self):
        self.learned = False
        self.optimal = None
        
    def fit(self, trainX,trainy):
        """
        
        """
        
        # Grid Search 
        skf = StratifiedKFold(n_splits=5)
        folds = list(skf.split(trainX, trainy))
        pipe = Pipeline([
            ("pca", PCA(random_state=777)),
            ("lda", LatentDirichletAllocation()),
            ("umap", umap(random_state=456)),
            ("clf",OneClassSVM())
            ])
        
        grid_params = [{'pca__n_components':[0.8, 2, 4,8],'pca__svd_solver':['full','arpack','randomized'],
                        'lda':['passthrough'],'umap':['passthrough'],
                        'clf__kernel': ['rbf','sigmoid'],'clf__gamma':[0.001,0.01,0.05,0.5],
                        'clf__nu':[0.001,0.01,0.03,0.5]},
                       {'lda__n_components': [10, 15, 20, 25, 30], 'lda__learning_decay': [.5, .7, .9],
                         'pca':['passthrough'],'umap':['passthrough'],
                        'clf__kernel': ['rbf','sigmoid'],'clf__gamma':[0.001,0.01,0.05,0.5],
                        'clf__nu':[0.001,0.01,0.03,0.5]},
                       {'umap__n_neighbors':[0.3,3,6,12,20],'umap__n_components':[0.8,2,4,8],
                        'pca':['passthrough'],'lda':['passthrough'],
                        'clf__kernel': ['rbf','sigmoid'],'clf__gamma':[0.001,0.01,0.05,0.5],
                        'clf__nu':[0.001,0.01,0.03,0.5]}]
        rocaucsc = make_scorer(roc_auc_score)
        search = GridSearchCV(pipe, grid_params, scoring=rocaucsc, cv=folds)
        search.fit(trainX, trainy)
        
        self.optimal = search.best_estimator_
        self.learned = True
        
        return self
        
    def predict(self, testX):
        """
        
        """
        if not self.learned:
            raise NameError('Fit model first')
            
        pred = self.optimal.predict(testX)
        return pred

    


