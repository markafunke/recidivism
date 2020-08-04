#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 21:21:24 2020

@author: mark
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

#Set up dataframes
compas_df = pd.read_pickle("pickle.p")
from sklego.preprocessing import InformationFilter
compas_df["race_black"] = compas_df["race"].apply(lambda x: 1 if x == "African-American" else 0)
compas_df["race_white"] = compas_df["race"].apply(lambda x: 1 if x == "Caucasian" else 0)
compas_df["felony"] = compas_df["c_charge_degree"].apply(lambda x: 0 if x == "M" else 1)
compas_df["female"] = compas_df["sex"].apply(lambda x: 0 if x == "Male" else 1)
mask = (compas_df["race_black"] == 1) | (compas_df["race_white"] == 1)
compas_wb = compas_df[mask]
info_df = compas_wb[["priors_count","age","race_black","two_year_recid"]]
X = info_df.drop("two_year_recid",axis=1)
y = info_df.two_year_recid

mod1 = LogisticRegression()
fairness_cv(mod1,X,y)
mod2 = GradientBoostingClassifier(learning_rate = .1)

column_list = ["names","auc_all", "acc_all", "prec_all", "rec_all", "fb_all", "pr_all" 
    , "auc_white", "acc_white", "prec_white", "rec_white", "fb_white", "pr_white"
    , "auc_black", "acc_black", "prec_black", "rec_black", "fb_black", "pr_black", "p_percent"]



model_list = [mod1,mod2,mod3,mod4,mod5,mod6,mod7,mod8]
model_names = 

mod1 = LogisticRegression(C=.001)
mod2 = KNeighborsClassifier(n_neighbors = 20)
mod3 = GradientBoostingClassifier(learning_rate = .001, max_depth = 6, n_estimators = 300)
mod4 = RandomForestClassifier()
mod5 = GaussianNB()


out1 = fairness_cv(mod1,X,y,name = 'log_fair', fairness = True)
out2 = fairness_cv(mod1,X,y,name = 'log_unfair', fairness = False)
out3 = fairness_cv(mod2,X,y,name = 'KNN_fair', fairness = True)
out4 = fairness_cv(mod2,X,y,name = 'KNN_unfair', fairness = False)
out5 = fairness_cv(mod3,X,y,name = 'GB_fair', fairness = True)
out6 = fairness_cv(mod3,X,y,name = 'GB_unfair', fairness = False)
out7 = fairness_cv(mod4,X,y,name = 'RF_fair', fairness = True)
out8 = fairness_cv(mod4,X,y,name = 'RF_unfair', fairness = False)
out9 = fairness_cv(mod5,X,y,name = 'NB_fair', fairness = True)
out10 = fairness_cv(mod5,X,y,name = 'NB_unfair', fairness = False)

output_all = pd.concat([out1,out2,out3,out4,out5,out6,out7,out8,out9,out10],axis=0)

summary = output_all.groupby("names").mean()
summary_trans = summary.transpose()

def fairness_cv(estimator,X,y,name="model_x",fairness = True):
    # Prepare data for cross validation
    X, y = np.array(X), np.array(y)
    kf = KFold(n_splits=5, shuffle=True, random_state = 123)
    
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

def score_b_w_std(log_fair):
    # Split out test set 
    X, X_test, y, y_test = \
    (train_test_split(info_df.drop("two_year_recid", axis = 1)
                     ,info_df.two_year_recid
                     ,test_size=0.2))
    
    # Standard Scale
    Scaler = StandardScaler()
    X_scale1 = Scaler.fit_transform(X)
    X_test_scale1 = Scaler.transform(X_test)
   
    # Filter out racial bias
    FairFilter = InformationFilter(len(X_scale1.T)-1)
    X_scaled = FairFilter.fit_transform(X_scale1)
    X_test_scaled = FairFilter.transform(X_test_scale1)
    
    # X_scaled = X
    # X_test_scaled = X_test
    
    # log_fair = LogisticRegression()
    log_fair.fit(X_scaled,y)
    soft_pred = log_fair.predict_proba(X_test_scaled)[:,1]
    hard_pred = log_fair.predict(X_test_scaled)
    
    # Calc metrics based for All, White, Black
    # All
    metrics.recall_score(y_test, hard_pred)
    
    # White
    mask_white = X_test["race_black"] == 0
    y_test_w = y_test[mask_white]
    y_pred_w = hard_pred[mask_white]
    p_percent_w = sum(y_pred_w)/len(y_pred_w)
    # print(metrics.accuracy_score(y_test_w, y_pred_w))
    # print(metrics.precision_score(y_test_w, y_pred_w))
    print(f"White acc: {metrics.accuracy_score(y_test_w, y_pred_w)}") 
    print(f"White precision: {metrics.precision_score(y_test_w, y_pred_w)}")    
    print(f"White recall: {metrics.recall_score(y_test_w, y_pred_w)}")
    print(f"White p_percent {p_percent_w}")
    
    # Black
    y_test_b = y_test[~mask_white]
    y_pred_b = hard_pred[~mask_white]
    p_percent_b = sum(y_pred_b)/len(y_pred_b)
    # print(metrics.accuracy_score(y_test, hard_pred))
    # print(metrics.precision_score(y_test_b, y_pred_b))
    print(f"Black acc {metrics.accuracy_score(y_test_b, y_pred_b)}")
    print(f"Black prec {metrics.precision_score(y_test_b, y_pred_b)}")
    print(f"Black recall {metrics.recall_score(y_test_b, y_pred_b)}")
    print(f"Black p_percent {p_percent_b}")
    
    print(f"AUC: {metrics.roc_auc_score(y_test, soft_pred)}")
    
    return p_percent_w, p_percent_b