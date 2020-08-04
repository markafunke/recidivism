#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:00:25 2020

@author: mark
"""
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer

# Define fbeta scorer, give 1/3 weight to recall vs precision
fb_scorer = make_scorer(fbeta_score, beta=1/3)

# Find best parameters for Logistic Regression
params={'model__C': [.0001,.001,.01,.1,1],
           'model__penalty':['l1','l2']}

pipeline = Pipeline([
    ('std_scale', StandardScaler()),
    ('info_scale', InformationFilter(2)),
    ('model',LogisticRegression())])

grid = GridSearchCV(pipeline, cv=5, n_jobs=-1, param_grid=params ,scoring=fb_scorer)
grid.fit(X,y)
grid.best_score_
grid.best_params_


# Find best parameters for KNN
params={'model__n_neighbors': np.linspace(10,500,50,dtype=int),
           'model__weights':['uniform','distance']}

pipeline = Pipeline([
    ('std_scale', StandardScaler()),
    ('info_scale', InformationFilter(2)),
    ('model',KNeighborsClassifier())])

grid = GridSearchCV(pipeline, cv=5, n_jobs=-1, param_grid=params ,scoring=fb_scorer)
grid.fit(X,y)
grid.best_score_
grid.best_params_

# Find best parameters for Gradient Boost
params={'model__learning_rate': np.linspace(0.001,1,20)
        , 'model__n_estimators':np.linspace(50,500,10,dtype=int)
        , 'model__max_depth':[3,4,5,6]}

pipeline = Pipeline([
    ('std_scale', StandardScaler()),
    ('info_scale', InformationFilter(2)),
    ('model',GradientBoostingClassifier())])

grid = GridSearchCV(pipeline, cv=5, n_jobs=-1, param_grid=params ,scoring=fb_scorer)
grid.fit(X,y)
grid.best_score_
grid.best_params_


