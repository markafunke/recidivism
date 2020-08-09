#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fits and evaluates recidivism classification models both pre and post 
transforming the data for fairness. 

Compares predictions for White and Black defendants to evaluate racial bias
,and exports the final predictions for use in a Tableau demonstration.

@author: markafunke
"""

import numpy as np
from classification_util import fairness_cv, fairness_train_test, eval_compas_scores, all_thresholds_df
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataframe from EDA preprocessing
compas_df = pd.read_pickle("pickles/compas_post_eda.p")

# Limit dataframe to just White/Black for simple binary comparison
# Could expand to look at fairness of other races in the future
mask = (compas_df["race_black"] == 1) | (compas_df["race_white"] == 1)
compas_wb = compas_df[mask]

# test COMPAS decile scores predictive accuracy as a base model
# values are scored 1-10, and I am interpreting as a % likelihood of recividism
# testing on full dataset as this county can be considered their full "test set"
# The most important metrics to me are 
#   F(1/3): which weights precision 3x more than recall. Chose this weighting
#   because the justice system is built on innocent until proven guilty
#   which means we want to limit false positives.
#   p_percent: The ratio of Black positive prediction % to White positive prediction %
#   A score close to 1.0 means the model is not biased
# Findings: F(1/3) score of .6506, p_percent of .497
white_df = compas_df[compas_df["race_white"] == 1]
black_df = compas_df[compas_df["race_black"] == 1]

conf_all = eval_compas_scores(compas_df)
conf_white = eval_compas_scores(white_df)
conf_black = eval_compas_scores(black_df)

# Choose features to use in analysis
# This is an iterative process, can change model df to test different features
# Note that "race_black" has to be the final column in X for the rest of the
# analysis to work
model_df = compas_wb[["priors_count","age","felony","female","race_black","two_year_recid"]]
X = model_df.drop("two_year_recid",axis=1)
y = model_df.two_year_recid

# Split train/test sets. Set random state for replicability
X_train, X_test, y_train, y_test = \
(train_test_split(X,y ,test_size=0.2, random_state=34, stratify=y))

X_train.to_pickle("pickles/X_train.p")
y_train.to_pickle("pickles/y_train.p")

# Choose models to test both with and without information filter
# Note that this is an iterative process, and the parameters below represent
# the final parameters ultimately chosen based on hyperparameter tuning in the
# "hyperparameter_tuning.py" file.
# Ultimately Logistic Regression has the best balance of high score in both
# fbeta and p_percent, and is my final model

mod1 = LogisticRegression(C=.001, solver='lbfgs',penalty='l2')
mod2 = KNeighborsClassifier(n_neighbors = 30, weights = 'uniform')
mod3 = XGBClassifier(learning_rate = .0536, max_depth = 4, n_estimators = 150 )
mod4 = RandomForestClassifier()
mod5 = GaussianNB()

# Run each model with and without "fairness" filter on, save to DataFrame
# in order to evaluate metrics
out1 = fairness_cv(mod1,X_train,y_train,name = 'log_fair', fairness = True)
out2 = fairness_cv(mod1,X_train,y_train,name = 'log_unfair', fairness = False)
out3 = fairness_cv(mod2,X_train,y_train,name = 'KNN_fair', fairness = True)
out4 = fairness_cv(mod2,X_train,y_train,name = 'KNN_unfair', fairness = False)
out5 = fairness_cv(mod3,X_train,y_train,name = 'GB_fair', fairness = True)
out6 = fairness_cv(mod3,X_train,y_train,name = 'GB_unfair', fairness = False)
out7 = fairness_cv(mod4,X_train,y_train,name = 'RF_fair', fairness = True)
out8 = fairness_cv(mod4,X_train,y_train,name = 'RF_unfair', fairness = False)
out9 = fairness_cv(mod5,X_train,y_train,name = 'NB_fair', fairness = True)
out10 = fairness_cv(mod5,X_train,y_train,name = 'NB_unfair', fairness = False)

output_all = pd.concat([out1,out2,out3,out4,out5,out6,out7,out8,out9,out10],axis=0)

summary = output_all.groupby("names").mean()
summary_trans = summary.transpose()

# Test final Logistic Model on test set
# Findings: As seen in summary_trans_test
    # Fair f(1/3): .662 (compared to .698 in "unfair" model amd .651 in COMPAS model)
    # Fair p_percent: .869 (improved from just .171 in "unfair" model and .497 in COMPAS model)

final_model = LogisticRegression(C=.001, solver='lbfgs',penalty='l2')
final_scores = fairness_train_test(final_model,X,y,fairness = True, name = "fair")
final_scores_unfair = fairness_train_test(final_model,X,y,fairness = False, name = "unfair")

output_all = pd.concat([final_scores,final_scores_unfair],axis=0)

summary_test = output_all.groupby("name").mean()
summary_trans_test = summary_test.transpose()

# Create csv of predictions on entire set to use in Tableau demonstration
# While ultimately testing my finding on the "test" set, for the purpose
# of demonstration, and wanting to compare predicitons 1to1 vs COMPAS,
# exporting a prediction for every defendant in the set

#1) COMPAS Predictions
# Making assumption 1-10 decile score is equivalent to percent likelihood
soft_pred_compas = compas_wb["decile_score"] / 10
compas_thresh = all_thresholds_df(compas_wb,soft_pred_compas, "COMPAS")

#2) Fair Predictions
Scaler = StandardScaler()
X_std_scale = Scaler.fit_transform(X_train)
X_test_std_scale = Scaler.transform(X)

FairFilter = InformationFilter(4) #scales on last column (race_black)
X_scaled = FairFilter.fit_transform(X_std_scale)
X_test_scaled = FairFilter.transform(X_test_std_scale)

final_model.fit(X_scaled,y_train)
soft_pred_fair = final_model.predict_proba(X_test_scaled)[:,1]

fair_thresh = all_thresholds_df(compas_wb,soft_pred_fair, "Fair")

#3) Unfair Predictions
Scaler = StandardScaler()
X_std_scale = Scaler.fit_transform(X_train)
X_test_std_scale = Scaler.transform(X)

final_model.fit(X_std_scale,y_train)
soft_pred_unfair = final_model.predict_proba(X_test_std_scale)[:,1]

unfair_thresh = all_thresholds_df(compas_wb,soft_pred_unfair, "Unfair")

#4) Concatenate and Export
final_tableau = pd.concat([compas_thresh,fair_thresh,unfair_thresh], axis = 0)
final_tableau.to_csv("tableau_data.csv")


