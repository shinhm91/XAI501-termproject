#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:16:12 2021

@author: juliashin
"""
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import umap # pip install umap-learn

# -------------------------------------
# Load preprocessed data
# -------------------------------------

class isoforest():
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
            ("clf",IsolationForest())
            ])
        
        grid_params = [{'pca__n_components':[0.8, 2, 4,8],'pca__svd_solver':['full','arpack','randomized'],
                        'lda':['passthrough'],'umap':['passthrough'],
                        'clf__n_estimators': [55, 75, 95, 115]},
                       {'lda__n_components': [10, 15, 20, 25, 30], 'lda__learning_decay': [.5, .7, .9],
                         'pca':['passthrough'],'umap':['passthrough'],
                        'clf__n_estimators': [55, 75, 95, 115]},
                       {'umap__n_neighbors':[0.3,3,6,12,20],'umap__n_components':[0.8,2,4,8],
                        'pca':['passthrough'],'lda':['passthrough'],
                        'clf__n_estimators': [55, 75, 95, 115]}]
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
