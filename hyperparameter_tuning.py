#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses GridSearch to fit the best models on a target metric of F(1/3)
These models are ultimately used in the "modeling.py" file

@author: mark
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklego.preprocessing import InformationFilter
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Read in X_train and y_train
X = pd.read_pickle("pickles/X_train.p")
y = pd.read_pickle("pickles/y_train.p")

# Define fbeta scorer, give 1/3 weight to recall vs precision
fb_scorer = make_scorer(fbeta_score, beta=1/3)

# Find best parameters for Logistic Regression
params={'model__C': [.0001,.001,.01,.1,1],
           'model__penalty':['l1','l2'],
           'model__solver': ['lbfgs','liblinear']}

pipeline = Pipeline([
    ('std_scale', StandardScaler()),
    ('info_scale', InformationFilter(4)),
    ('model',LogisticRegression())])

grid = GridSearchCV(pipeline, cv=5, n_jobs=-1, param_grid=params ,scoring=fb_scorer)
grid.fit(X,y)
grid.best_score_
grid.best_params_


# Find best parameters for KNN
params={'model__n_neighbors': np.linspace(10,500,50,dtype=int)
        ,'model__weights':['uniform','distance']}

pipeline = Pipeline([
    ('std_scale', StandardScaler()),
    ('info_scale', InformationFilter(4)),
    ('model',KNeighborsClassifier())])

grid = GridSearchCV(pipeline, cv=5, n_jobs=-1, param_grid=params ,scoring=fb_scorer)
grid.fit(X,y)
grid.best_score_
grid.best_params_

# Find best parameters for Gradient Boost
params={'model__learning_rate': np.linspace(0.001,1,20)
        , 'model__n_estimators':np.linspace(50,500,10,dtype=int)
        , 'model__max_depth':[4,5,6,7,8]}

pipeline = Pipeline([
    ('std_scale', StandardScaler()),
    ('info_scale', InformationFilter(4)),
    ('model',XGBClassifier())])

grid = GridSearchCV(pipeline, cv=5, n_jobs=-1, param_grid=params ,scoring=fb_scorer)
grid.fit(X,y)
grid.best_score_
grid.best_params_


