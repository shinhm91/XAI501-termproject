#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:16:12 2021

@author: juliashin
"""
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score,make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV

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
        clf = IsolationForest()        
        clf_params = {'contamination': np.linspace(0.0, 0.05, 15),'n_estimators': [55, 75, 95, 115]}
        
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
