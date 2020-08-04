#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:49:22 2020

@author: markfunke
"""
from sqlalchemy import create_engine
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

#connect to SQL database
# cnx = create_engine('postgresql://ubuntu@18.191.18.59:5432/baseball')
# compas_df = pd.read_sql_query('''SELECT * FROM compastwoyear''', cnx)
compas_df = pd.read_pickle("pickle.p")

# check totals
#implies .545 accuracy just guessing 0, relatively balanced
compas_df["two_year_recid"].value_counts()

# check for any big differences in average recidivism by feature
# there are pretty substantial differences in all crime counts
# including juvenile misdemeanors, felones, other, and total prior counts
# there is also a 4 year gap showing younger people tend to reoffend, on average
average_recid = compas_df.groupby('two_year_recid').mean()
average_by_race = compas_df.groupby('race').mean()

# there appears to be a jump in reoffending for > 0 juveniles
# create boolean features for both juvenile misdemeanors and felonies
compas_df["juv_m_gt1"] = compas_df["juv_misd_count"].apply(lambda x: 0 if x == 0 else 1)
compas_df["juv_f_gt1"] = compas_df["juv_fel_count"].apply(lambda x: 0 if x == 0 else 1)
compas_df["felony"] = compas_df["c_charge_degree"].apply(lambda x: 0 if x == "M" else 1)
X["female"] = X["sex"].apply(lambda x: 0 if x == "Male" else 1)

# create boolean to test results by race
compas_df.race.value_counts()
compas_df["race_black"] = compas_df["race"].apply(lambda x: 1 if x == "African-American" else 0)
compas_df["race_white"] = compas_df["race"].apply(lambda x: 1 if x == "Caucasian" else 0)

# split out final testing dataset
# set random state for reproducibility
X, X_test, y, y_test = \
(train_test_split(compas_df.drop("two_year_recid", axis = 1)
                 ,compas_df.two_year_recid
                 ,test_size=0.2, random_state=34))

# test COMPAS decile scores predictive accuracy
# values are scored 1-10, and I am interpreting as a % likelihood of recividism
# testing on full dataset as this county can be considered their full "test set"
white_df = compas_df[compas_df["race_white"] == 1]
black_df = compas_df[compas_df["race_black"] == 1]

conf_all = eval_compas_scores(compas_df)
conf_white = eval_compas_scores(white_df)
conf_black = eval_compas_scores(black_df)

# EDA - check priors count by race
pd.crosstab(white_df.priors_count, white_df.two_year_recid.astype(bool)).plot(kind='bar')
plt.title('Recidivism by Priors Count - White')
plt.xlabel('Priors')
plt.ylabel('Frequency')
plt.ylim(0,600)
plt.xlim(-1,20);

pd.crosstab(black_df.priors_count, black_df.two_year_recid.astype(bool)).plot(kind='bar')
plt.title('Recidivism by Priors Count - Black')
plt.xlabel('Priors')
plt.ylabel('Frequency')
plt.ylim(0,600)
plt.xlim(-1,20);

X_arr = np.array(X.iloc[:,0:8])
mask = [X_arr[:,-1] == "Caucasian"]
test = X_arr[mask]

test = X_arr[:,:-1] 

# base model
base = DummyClassifier()
cross_val_scores(base, X, y, name = 'base')



# model 1
model1 = LogisticRegression()
cross_val_race_info(model1, X[["priors_count","age","female","race_black"]] , y)
cross_val_scores_std(model1, X[["priors_count","age","sex","race_black"]] , y)
cross_val_race(model1, X[["priors_count","age","felony","race"]] , y)

predictions = model1.predict(X_fair[["priors_count","age","female"]])
predictions_proba = model1.predict_proba(X_fair[["priors_count","age","female"]])
X["predict"] = predictions
X["predict_proba"] = predictions_proba[:,1]
X["actual"] = y
X.to_csv("explore_predict4.csv")
X_fair.to_csv("explore_predict5.csv")

# from sklego.metrics import p_percent_score
# model1 = LogisticRegression(solver='lbfgs').fit(X[["priors_count","age","female","race_black"]], y)
# print('p_percent_score:', p_percent_score(sensitive_column="race_black")(model1, X[["priors_count","age","female","race_black"]],y))

# model_comp = LogisticRegression(solver='lbfgs').fit(X[["decile_score","race_black"]], y)
# print('p_percent_score:', p_percent_score(sensitive_column="race_black")(model_comp, X[["decile_score","race_black"]],y))




# Test  Information Filter Logic

#Set up dataframes
compas_df = pd.read_pickle("pickle.p")
from sklego.preprocessing import InformationFilter
compas_df["race_black"] = compas_df["race"].apply(lambda x: 1 if x == "African-American" else 0)
compas_df["race_white"] = compas_df["race"].apply(lambda x: 1 if x == "Caucasian" else 0)
compas_df["felony"] = compas_df["c_charge_degree"].apply(lambda x: 0 if x == "M" else 1)
compas_df["female"] = compas_df["sex"].apply(lambda x: 0 if x == "Male" else 1)
mask = (compas_df["race_black"] == 1) | (compas_df["race_white"] == 1)
compas_wb = compas_df[mask]
info_df = compas_wb[["priors_count","age","felony","race_black","two_year_recid"]]


mod1 = LogisticRegression()
mod2 = RandomForestClassifier()
mod3 = GradientBoostingClassifier(learning_rate = .01)
from sklego.linear_model import DemographicParityClassifier
mod4 = DemographicParityClassifier(sensitive_cols=3, covariance_threshold=0)


score_b_w(mod1)
score_b_w(mod2)
score_b_w(mod3)
score_b_w(mod4)
score_b_w(base)



score_b_w_std(mod1)
score_b_w_std(mod2)

score_b_w_std(mod3)
score_b_w(mod3)

score_b_w_std(mod4)

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

def score_b_w(log_fair):
    # Split out test set 
    X, X_test, y, y_test = \
    (train_test_split(info_df.drop("two_year_recid", axis = 1)
                     ,info_df.two_year_recid
                     ,test_size=0.2))
    
    # # Filter out racial bias
    # FairFilter = InformationFilter(["race_black"])
    # X_scaled = FairFilter.fit_transform(X)
    # X_test_scaled = FairFilter.transform(X_test)
    
    
    # Standard Scale
    Scaler = StandardScaler()
    X_scaled = Scaler.fit_transform(X)
    X_test_scaled = Scaler.transform(X_test)

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








fair_classifier = DemographicParityClassifier(sensitive_cols="race_black", covariance_threshold=0.1)
fair_classifier.fit(X[["priors_count","age","female","race_black"]], y);
preds = fair_classifier.predict(X[["priors_count","age","female","race_black"]])
metrics.accuracy_score(y, preds)
metrics.precision_score(y, preds)
metrics.recall_score(y, preds)
X["pred"] = preds
X["actual"] = y
X.to_csv("testestetst2.csv")


# InformationFilter(["race_black"]).transform(test_df)

model10 = LogisticRegression()
model10.fit(X_fair,y)
cross_val_scores(model10, X_fair, y)

X_fair["race"] = X["race"]
cross_val_race(model1, X_fair, y)


predictions = model1.predict(X_fair[["priors_count","age","female"]])
metrics.precision_score(y,predictions)
metrics.recall_score(y,predictions)

predictions_proba = model1.predict_proba(X_fair[["priors_count","age","female"]])
test_df["predict"] = predictions
test_df["predict_proba"] = predictions_proba[:,1]
test_df["actual"] = y
test_df.to_csv("explore_predict4.csv")
X_fair.to_csv("explore_predict5.csv")


# model 2
X["juv_m_gt1"] = X["juv_misd_count"].apply(lambda x: 0 if x == 0 else 1)
X["juv_f_gt1"] = X["juv_fel_count"].apply(lambda x: 0 if x == 0 else 1)
X["felony"] = X["c_charge_degree"].apply(lambda x: 0 if x == "M" else 1)
X["female"] = X["sex"].apply(lambda x: 0 if x == "Male" else 1)

X_mod2 = X[["priors_count", "age_cat", "felony","juv_m_gt1","juv_f_gt1","female"]]
X_mod2 = pd.get_dummies(X_mod2)


model2 = RandomForestClassifier()
cross_val_scores(model2, X_mod2 , y)
cross_val_race(model2, X[["priors_count", "age","race"]] , y)

model2.fit(X_mod2,y)
importances = model2.feature_importances_
importances

# model 3 
model3 = DecisionTreeClassifier()
cross_val_scores(model3, X[["priors_count", "age"]] , y)
cross_val_race(model3, X[["priors_count", "age","race"]] , y)

# model 4
model4 = GaussianNB()
cross_val_scores(model4, X_mod2 , y)
cross_val_race(model4, X[["priors_count", "age","race"]] , y)

# model 5
model5 = SVC()
cross_val_scores_std(model5, X[["priors_count", "age"]] , y)
cross_val_race(model5, X[["priors_count", "age","race"]] , y)

# model 6
model6 = KNeighborsClassifier(n_neighbors=150)
cross_val_scores_std(model6, X[["priors_count", "age"]] , y)
cross_val_race(model6, X[["priors_count", "age","race_white","race"]] , y)

# model 7
model7 = GradientBoostingClassifier(n_estimators = 100, learning_rate = .1)
cross_val_scores(model7, X[["priors_count", "age"]] , y)
cross_val_race(model7, X[["priors_count", "age","race"]] , y)

model7 = model7.fit(X_mod2,y)
plot_importance(model6)
plt.(show)

average_mod = compas_df.groupby('two_year_recid').mean()
compas_df.c_charge_desc.value_counts()




def cross_val_race_info(model, X, y, rand = 122, name = 'model1'):
    
    # Prepare data for cross validation
    X, y = np.array(X), np.array(y)
    kf = KFold(n_splits=5, shuffle=True, random_state = rand)
    
    # Initiate lists to store results
    cv_AUCs, cv_acc, cv_prec, cv_rec, cv_fb  = [], [], [], [], []
    cv_AUCs_w, cv_acc_w, cv_prec_w, cv_rec_w, cv_fb_w  = [], [], [], [], []
    cv_AUCs_b, cv_acc_b, cv_prec_b, cv_rec_b, cv_fb_b  = [], [], [], [], []

    
    for train_ind, val_ind in kf.split(X,y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]
        
        # Feature scaling - Information Scaler to make columns orthogonal to race
        scaler = InformationFilter(len(X_val.T)-1)
        X_train = scaler.fit_transform(X_train) #scale & drop race columm
        X_val = scaler.transform(X_val) #sclae & drop race columm
        
        # fit model and make predictions
        model.fit(X_train[:,:-1], y_train)
        soft_pred = model.predict_proba(X_val[:,:-1])[:,1]
        hard_pred = model.predict(X_val[:,:-1])
        
        # score all
        cv_AUCs.append(metrics.roc_auc_score(y_val, soft_pred))
        cv_acc.append(metrics.accuracy_score(y_val, hard_pred))
        cv_prec.append(metrics.precision_score(y_val, hard_pred))
        cv_rec.append(metrics.recall_score(y_val, hard_pred))
        cv_fb.append(metrics.fbeta_score(y_val, hard_pred, beta = 1/2))
        
        # score white
        mask_val_w = [X_val[:,-1] == 0]
        X_val_w = X_val[mask_val_w]
        X_val_w = X_val_w[:,:-1] #drop race columm
        y_val_w = y_val[mask_val_w]
        
        soft_pred = model.predict_proba(X_val_w)[:,1]
        hard_pred = model.predict(X_val_w)
        
        # score all
        cv_AUCs_w.append(metrics.roc_auc_score(y_val_w, soft_pred))
        cv_acc_w.append(metrics.accuracy_score(y_val_w, hard_pred))
        cv_prec_w.append(metrics.precision_score(y_val_w, hard_pred))
        cv_rec_w.append(metrics.recall_score(y_val_w, hard_pred))
        cv_fb_w.append(metrics.fbeta_score(y_val_w, hard_pred, beta = 1/2))               
        
        # score black
        mask_val_b = [X_val[:,-1] == 1]
        X_val_b = X_val[mask_val_b]
        X_val_b = X_val_b[:,:-1] #drop race columm
        y_val_b = y_val[mask_val_b]
        
        soft_pred = model.predict_proba(X_val_b)[:,1]
        hard_pred = model.predict(X_val_b)
        
        # score all
        cv_AUCs_b.append(metrics.roc_auc_score(y_val_b, soft_pred))
        cv_acc_b.append(metrics.accuracy_score(y_val_b, hard_pred))
        cv_prec_b.append(metrics.precision_score(y_val_b, hard_pred))
        cv_rec_b.append(metrics.recall_score(y_val_b, hard_pred))
        cv_fb_b.append(metrics.fbeta_score(y_val_b, hard_pred, beta = 1/2)) 
        
        # Feature scaling
        # Only used by KNN
        #scaler = StandardScaler()
        #X_train_scaled = scaler.fit_transform(X_train)
        #X_val_scaled = scaler.transform(X_val)
        
    model_AUCs = round(np.mean(cv_AUCs),3)
    model_acc = round(np.mean(cv_acc),3)
    model_prec = round(np.mean(cv_prec),3)
    model_rec = round(np.mean(cv_rec),3)
    model_fb = round(np.mean(cv_fb),3)
    
    print(f"{name} AUC: {model_AUCs}")
    print(f"{name} Acc: {model_acc}")
    print(f"{name} Prec: {model_prec}")
    print(f"{name} Rec: {model_rec}")
    print(f"{name} Fbeta: {model_fb}")

    model_AUCs_w = round(np.mean(cv_AUCs_w),3)
    model_acc_w = round(np.mean(cv_acc_w),3)
    model_prec_w = round(np.mean(cv_prec_w),3)
    model_rec_w = round(np.mean(cv_rec_w),3)
    model_fb_w = round(np.mean(cv_fb_w),3)
    
    print(f"{name} AUC w: {model_AUCs_w}")
    print(f"{name} Acc w: {model_acc_w}")
    print(f"{name} Prec w: {model_prec_w}")
    print(f"{name} Rec w: {model_rec_w}")
    print(f"{name} Fbeta w: {model_fb_w}")
    
    model_AUCs_b = round(np.mean(cv_AUCs_b),3)
    model_acc_b = round(np.mean(cv_acc_b),3)
    model_prec_b = round(np.mean(cv_prec_b),3)
    model_rec_b = round(np.mean(cv_rec_b),3)
    model_fb_b = round(np.mean(cv_fb_b),3)
    
    print(f"{name} AUC b: {model_AUCs_b}")
    print(f"{name} Acc b: {model_acc_b}")
    print(f"{name} Prec b: {model_prec_b}")
    print(f"{name} Rec b: {model_rec_b}")
    print(f"{name} Fbeta b: {model_fb_b}")




def eval_compas_scores(df):
    y_actual = df[["two_year_recid"]]
    y_soft_guess = df["decile_score"] / 10
    print(metrics.roc_auc_score(y_actual, y_soft_guess))
    
    y_hard_guess = df["decile_score"].apply(lambda x: 1 if x > 5 else 0)
    print(metrics.accuracy_score(y_actual, y_hard_guess))
    print(metrics.precision_score(y_actual, y_hard_guess))
    print(metrics.recall_score(y_actual, y_hard_guess))
    print(metrics.fbeta_score(y_actual, y_hard_guess, beta = 1/2))
    
    return metrics.confusion_matrix(y_actual, y_hard_guess)



def cross_val_scores(model, X, y, rand = 122, name = 'model1'):
    
    # Prepare data for cross validation
    X, y = np.array(X), np.array(y)
    kf = KFold(n_splits=5, shuffle=True, random_state = rand)
    
    # Initiate lists to store results
    cv_AUCs = []
    cv_acc = []
    cv_prec = []
    cv_rec = []
    cv_fb = []
    
    for train_ind, val_ind in kf.split(X,y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]
        
        # Create Classifier Object
        # model = classifier
        
        # fit model and make predictions
        model.fit(X_train, y_train)
        soft_pred = model.predict_proba(X_val)[:,1]
        hard_pred = model.predict(X_val)
        
        # score model
        cv_AUCs.append(metrics.roc_auc_score(y_val, soft_pred))
        cv_acc.append(metrics.accuracy_score(y_val, hard_pred))
        cv_prec.append(metrics.precision_score(y_val, hard_pred))
        cv_rec.append(metrics.recall_score(y_val, hard_pred))
        cv_fb.append(metrics.fbeta_score(y_val, hard_pred, beta = 1/2))
        
        
        # Feature scaling
        # Only used by KNN / SVM
        #scaler = StandardScaler()
        #X_train_scaled = scaler.fit_transform(X_train)
        #X_val_scaled = scaler.transform(X_val)
        
    model_AUCs = round(np.mean(cv_AUCs),3)
    model_acc = round(np.mean(cv_acc),3)
    model_prec = round(np.mean(cv_prec),3)
    model_rec = round(np.mean(cv_rec),3)
    model_fb = round(np.mean(cv_fb),3)
    
    print(f"{name} AUC: {model_AUCs}")
    print(f"{name} Acc: {model_acc}")
    print(f"{name} Prec: {model_prec}")
    print(f"{name} Rec: {model_rec}")
    print(f"{name} Fbeta: {model_fb}")
 
    

    
def cross_val_scores_info_scaler(model, X, y, rand = 122, name = 'model1'):
    
    # Prepare data for cross validation
    X, y = np.array(X), np.array(y)
    kf = KFold(n_splits=5, shuffle=True, random_state = rand)
    
    # Initiate lists to store results
    cv_AUCs = []
    cv_acc = []
    cv_prec = []
    cv_rec = []
    cv_fb = []
    
    for train_ind, val_ind in kf.split(X,y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]
        
        # Feature scaling - Information Scaler to make columns orthogonal to race
        scaler = InformationFilter(len(X_val.T)-1)
        X_train_scaled = scaler.fit_transform(X_train)[:,:-1] #scale & drop race columm
        X_val_scaled = scaler.transform(X_val)[:,:-1] #sclae & drop race columm
        
        # fit model and make predictions
        model.fit(X_train_scaled, y_train)
        soft_pred = model.predict_proba(X_val_scaled)[:,1]
        hard_pred = model.predict(X_val_scaled)
        
        # score model
        cv_AUCs.append(metrics.roc_auc_score(y_val, soft_pred))
        cv_acc.append(metrics.accuracy_score(y_val, hard_pred))
        cv_prec.append(metrics.precision_score(y_val, hard_pred))
        cv_rec.append(metrics.recall_score(y_val, hard_pred))
        cv_fb.append(metrics.fbeta_score(y_val, hard_pred, beta = 1/2))
        

    model_AUCs = round(np.mean(cv_AUCs),3)
    model_acc = round(np.mean(cv_acc),3)
    model_prec = round(np.mean(cv_prec),3)
    model_rec = round(np.mean(cv_rec),3)
    model_fb = round(np.mean(cv_fb),3)
    
    print(f"{name} AUC: {model_AUCs}")
    print(f"{name} Acc: {model_acc}")
    print(f"{name} Prec: {model_prec}")
    print(f"{name} Rec: {model_rec}")
    print(f"{name} Fbeta: {model_fb}")
    

    
    