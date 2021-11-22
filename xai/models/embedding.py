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
from sklearn.metrics import roc_auc_score,make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV

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
        clf = OneClassSVM()        
        clf_params = {'kernel': ['rbf','sigmoid'],'gamma':[0.001,0.01,0.05,0.5],
                        'nu':[0.001,0.01,0.03,0.5]}
            
        rocaucsc = make_scorer(roc_auc_score)
        search = GridSearchCV(clf, clf_params, scoring=rocaucsc, cv=folds)
        search.fit(trainX, trainy)
        
        self.optimal = search.best_estimator_
        self.learned = True
        
        return self
    
    def get_cv_rocauc(self):
        return search.cv_results_['mean_test_score']
    
    def get_gridcv_params(self):
        return search.cv_results_['params']
        
    def predict(self, testX):
        """
        
        """
        if not self.learned:
            raise NameError('Fit model first')
            
        pred = self.optimal.predict(testX)
        return pred

    


