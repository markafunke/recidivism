#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains 3 functions used to score classification models
@author: markfunke
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklego.preprocessing import InformationFilter
from sklearn.model_selection import train_test_split

def fairness_cv(estimator,X,y,name="model_x",fairness = True):
    '''
    Runs cross validation on model given feature set and target variable.
    Output dataframe with classification metrics split by race.

    Parameters
    ----------
    estimator : Classification model object
        Save an instance of any classification model and input
    X : Array or DataFrame
        Feature set. Race column must be final column.
    y : Array or DataFrame
        Target variable.
    name : string, optional
        Name of model used in DataFrame. The default is "model_x".
    fairness : Boolean, optional
        True = Apply fairness transformation. The default is True.

    Returns
    -------
    output_df : DataFrame
        Returns DataFrame containing classification metrics split by total,
        Black, and White.

    '''
    # Prepare data for cross validation
    X, y = np.array(X), np.array(y)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 34)
    
    # Initiate lists to store results    
    auc_all, acc_all, prec_all, rec_all, fb_all, pr_all  = [], [], [], [], [], []
    auc_white, acc_white, prec_white, rec_white, fb_white, pr_white  = [], [], [], [], [], []
    auc_black, acc_black, prec_black, rec_black, fb_black, pr_black  = [], [], [], [], [],[]
    p_percent = []
    
    for train_ind, val_ind in kf.split(X,y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]

        # Standard Scale
        Scaler = StandardScaler()
        X_std_scale = Scaler.fit_transform(X_train)
        X_val_std_scale = Scaler.transform(X_val)
       
        if fairness:
        # Transform vectors to be orthogonal to race_black = 1
            FairFilter = InformationFilter(len(X_std_scale.T)-1) #scales on last column (race_black)
            X_scaled = FairFilter.fit_transform(X_std_scale)
            X_test_scaled = FairFilter.transform(X_val_std_scale)
        else:
            X_scaled = X_std_scale
            X_test_scaled = X_val_std_scale
        
        estimator.fit(X_scaled,y_train)
        soft_pred = estimator.predict_proba(X_test_scaled)[:,1]
        hard_pred = estimator.predict(X_test_scaled)
        pr = sum(hard_pred)/len(hard_pred)
        
        mask_white = X_val[:,-1] == 0
        y_val_w = y_val[mask_white]
        soft_pred_w = soft_pred[mask_white]
        hard_pred_w = hard_pred[mask_white]
        pr_w = sum(hard_pred_w)/len(hard_pred_w)
        
        y_val_b = y_val[~mask_white]
        soft_pred_b = soft_pred[~mask_white]
        hard_pred_b = hard_pred[~mask_white]
        pr_b = sum(hard_pred_b)/len(hard_pred_b)        
        
        # AUC
        auc_all.append(metrics.roc_auc_score(y_val, soft_pred))
        auc_white.append(metrics.roc_auc_score(y_val_w, soft_pred_w))
        auc_black.append(metrics.roc_auc_score(y_val_b, soft_pred_b))
        
        # Accuracy
        acc_all.append(metrics.accuracy_score(y_val, hard_pred))
        acc_white.append(metrics.accuracy_score(y_val_w, hard_pred_w))
        acc_black.append(metrics.accuracy_score(y_val_b, hard_pred_b))
        
        # Precision
        prec_all.append(metrics.precision_score(y_val, hard_pred))
        prec_white.append(metrics.precision_score(y_val_w, hard_pred_w))
        prec_black.append(metrics.precision_score(y_val_b, hard_pred_b))
        
        # Recall
        rec_all.append(metrics.recall_score(y_val, hard_pred))
        rec_white.append(metrics.recall_score(y_val_w, hard_pred_w))
        rec_black.append(metrics.recall_score(y_val_b, hard_pred_b))
        
        # Fbeta
        fb_all.append(metrics.fbeta_score(y_val, hard_pred, beta = 1/3))
        fb_white.append(metrics.fbeta_score(y_val_w, hard_pred_w, beta = 1/3))
        fb_black.append(metrics.fbeta_score(y_val_b, hard_pred_b, beta = 1/3))        
        
        # Positive Guess Rate
        pr_all.append(pr)
        pr_white.append(pr_w)
        pr_black.append(pr_b)
        
        # Positive Guess Rate Ratio
        p_percent_ratio = min(pr_b/pr_w, pr_w/pr_b)
        p_percent.append(p_percent_ratio)
        
        # Name model to use in DataFrame output
        names = [name]*5
        
    out_list = [names,auc_all, acc_all, prec_all, rec_all, fb_all, pr_all 
    , auc_white, acc_white, prec_white, rec_white, fb_white, pr_white
    , auc_black, acc_black, prec_black, rec_black, fb_black, pr_black, p_percent]
    
    column_list = ["names","auc_all", "acc_all", "prec_all", "rec_all", "fb_all", "pr_all" 
    , "auc_white", "acc_white", "prec_white", "rec_white", "fb_white", "pr_white"
    , "auc_black", "acc_black", "prec_black", "rec_black", "fb_black", "pr_black", "p_percent"]

    output_df = pd.DataFrame(np.array(out_list).T, columns= column_list)
    
    column_num = [n for n in output_df.columns if n not in ['names']]
    for column in column_num:
        output_df[column] = pd.to_numeric(output_df[column])
    
    return output_df

def fairness_train_test(estimator,X,y,fairness = True, name = "model"):
    '''
    Runs train/test on model given feature set and target variable.
    Output dataframe with classification metrics split by race.

    Parameters
    ----------
    estimator : Classification model object
        Save an instance of any classification model and input
    X : Array or DataFrame
        Feature set. Race column must be final column.
    y : Array or DataFrame
        Target variable.
    name : string, optional
        Name of model used in DataFrame. The default is "model_x".
    fairness : Boolean, optional
        True = Apply fairness transformation. The default is True.

    Returns
    -------
    output_df : DataFrame
        Returns DataFrame containing classification metrics split by total,
        Black, and White.

    '''
    # Prepare data for cfinal testing
    # Selecting random state to be consistent with intitial split for
    # cross validation testing, and for replicability
    X, X_test, y, y_test = \
    (train_test_split(X,y ,test_size=0.2, random_state=34, stratify=y))
    
    # Standard Scale
    Scaler = StandardScaler()
    X_std_scale = Scaler.fit_transform(X)
    X_test_std_scale = Scaler.transform(X_test)
   
    if fairness:
    # Transform vectors to be orthogonal to race_black = 1
        FairFilter = InformationFilter(len(X_std_scale.T)-1) #scales on last column (race_black)
        X_scaled = FairFilter.fit_transform(X_std_scale)
        X_test_scaled = FairFilter.transform(X_test_std_scale)
    else:
        X_scaled = X_std_scale
        X_test_scaled = X_test_std_scale
    
    estimator.fit(X_scaled,y)
    soft_pred = estimator.predict_proba(X_test_scaled)[:,1]
    hard_pred = estimator.predict(X_test_scaled)
    pr_all = sum(hard_pred)/len(hard_pred)
    
    mask_white = X_test["race_black"] == 0
    y_test_w = y_test[mask_white]
    soft_pred_w = soft_pred[mask_white]
    hard_pred_w = hard_pred[mask_white]
    pr_white = sum(hard_pred_w)/len(hard_pred_w)
    
    y_test_b = y_test[~mask_white]
    soft_pred_b = soft_pred[~mask_white]
    hard_pred_b = hard_pred[~mask_white]
    pr_black = sum(hard_pred_b)/len(hard_pred_b)        
    
    # AUC
    auc_all = metrics.roc_auc_score(y_test, soft_pred)
    auc_white = metrics.roc_auc_score(y_test_w, soft_pred_w)
    auc_black = metrics.roc_auc_score(y_test_b, soft_pred_b)
    
    # Accuracy
    acc_all = metrics.accuracy_score(y_test, hard_pred)
    acc_white = metrics.accuracy_score(y_test_w, hard_pred_w)
    acc_black = metrics.accuracy_score(y_test_b, hard_pred_b)
    
    # Precision
    prec_all = metrics.precision_score(y_test, hard_pred)
    prec_white = metrics.precision_score(y_test_w, hard_pred_w)
    prec_black = metrics.precision_score(y_test_b, hard_pred_b)
    
    # Recall
    rec_all = metrics.recall_score(y_test, hard_pred)
    rec_white = metrics.recall_score(y_test_w, hard_pred_w)
    rec_black = metrics.recall_score(y_test_b, hard_pred_b)
    
    # Fbeta
    fb_all = metrics.fbeta_score(y_test, hard_pred, beta = 1/3)
    fb_white = metrics.fbeta_score(y_test_w, hard_pred_w, beta = 1/3)
    fb_black = metrics.fbeta_score(y_test_b, hard_pred_b, beta = 1/3)    
    
    # Positive Guess Rate Ratio
    p_percent = min(pr_black/pr_white, pr_white/pr_black)

    
    out_list = [[name, auc_all, acc_all, prec_all, rec_all, fb_all, pr_all 
    , auc_white, acc_white, prec_white, rec_white, fb_white, pr_white
    , auc_black, acc_black, prec_black, rec_black, fb_black, pr_black, p_percent]]
    
    column_list = ["name","auc_all", "acc_all", "prec_all", "rec_all", "fb_all", "pr_all" 
    , "auc_white", "acc_white", "prec_white", "rec_white", "fb_white", "pr_white"
    , "auc_black", "acc_black", "prec_black", "rec_black", "fb_black", "pr_black", "p_percent"]

    output_df = pd.DataFrame(out_list, columns= column_list)
    
    column_num = [n for n in output_df.columns if n not in ['name']]
    for column in column_num:
        output_df[column] = pd.to_numeric(output_df[column])
    
    return output_df

def eval_compas_scores(df):
    '''
    Prints AUC, Accuracy, Precision, Recall, F(1/3) and Positive Guess Rate
    for COMPAS estimates. Given a dataframe containing decile score and 
    actual result (two_year_recid)

    Parameters
    ----------
    df : DataFrame
        dataframe containing decile score and 
        actual result (two_year_recid)
    '''
    y_actual = df[["two_year_recid"]]
    y_soft_guess = df["decile_score"] / 10
    y_hard_guess = df["decile_score"].apply(lambda x: 1 if x > 5 else 0)
    pr = sum(y_hard_guess)/len(y_hard_guess) 
    
    print(f"AUC: {metrics.roc_auc_score(y_actual, y_soft_guess)}")
    print(f"Accuracy: {metrics.accuracy_score(y_actual, y_hard_guess)}")
    print(f"Precision: {metrics.precision_score(y_actual, y_hard_guess)}")
    print(f"Recall: {metrics.recall_score(y_actual, y_hard_guess)}")
    print(f"F1/3: {metrics.fbeta_score(y_actual, y_hard_guess, beta = 1/3)}")
    print(f"Positive Rate: {pr}")
    
    
def all_thresholds_df(base_df, soft_pred, name):
    '''
    Creates 9 dataframes, 1 for each threshold .1 trhough .9.
    Intended to create dataframes with hard predictions by threshold to be
    used in Tableau Desktop dashboard.


    Parameters
    ----------
    base_df : DataFrame to append predictions to.
    soft_pred : Soft Prediction values from a classification model.
                Must be same length as base_df
    name : Name of model to be included in DataFrame output

    Returns
    -------
    final : DataFrame
        Outputs 9 base_df concatenated on top of eachother, one for each threshold
        Includes 4 additional columns - hard prediction, soft prediction, 
        threshold, and predictor (name of model)

    '''
    thresholds = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
    final = pd.DataFrame()
    for thresh in thresholds:
        results = base_df.copy()
        results["hard_pred"] = np.where(soft_pred > thresh, 1, 0)
        results["soft_pred"] = soft_pred
        results["treshold"] = thresh
        results["Predictor"] = name
        final = pd.concat([final,results],axis=0)
    return final