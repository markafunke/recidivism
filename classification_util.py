#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains 2 functions used to score linear regression models.
sm_summary - Stats Model linear fit summary
cross_val_score - Sklearn cross validation R2 score for both linear and Ridge.

@author: markfunke
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
    
def cross_val_auc_all(model, X, y, rand = 122):
    """
    For a set of features X, and target y, fit both Linear Regression
    and Ridge model. Validate with cross validation and print validation
    R2, RMSE,and print Ridge coefficients.

    Parameters
    ----------
    X : DataFrame of features
    y : Series of target variable
    rand : Integer to set random state. The default is None.
    lamb : Float to set lamda of Ridge model. The default is 1.

    Returns
    -------
    Ridge Model coefficients.

    """
    
    # Prepare data for cross validation
    X, y = np.array(X), np.array(y)
    kf = KFold(n_splits=5, shuffle=True, random_state = rand)
    
    # Initiate lists to store results
    cv_AUCs = []
    
    for train_ind, val_ind in kf.split(X,y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]
        
        # Create Classifier Object
#        model = classifier
        
        # linear model
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:,1]
        cv_AUCs.append(metrics.roc_auc_score(y_val, y_pred))
        
        # Feature scaling
        # Only used by KNN
        #scaler = StandardScaler()
        #X_train_scaled = scaler.fit_transform(X_train)
        #X_val_scaled = scaler.transform(X_val)
    model_AUCs = round(np.mean(cv_AUCs),3)
    print(f"Model AUC: {model_AUCs}")

if __name__ == '__main__':
    main()
    