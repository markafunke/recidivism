#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code includes the feature engineering and intuition behind the features 
ultimately tested and used in the modeling process.
Also includes exploratory plots used to inform the modeling process.

@author: markfunke
"""

#from sqlalchemy import create_engine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# connect to SQL database
# note I removed the AWS IP address for privacy, but if replicating this code,
# you can start with the pickle below
# cnx = create_engine('postgresql://ubuntu@XX.XXX.XX.XX:XXXX/compas')
# compas_df = pd.read_sql_query('''SELECT * FROM compastwoyear''', cnx)
# compas_df.to_pickle("compas_clean.p")
compas_df = pd.read_pickle("compas_clean.p")

# check totals
#implies .545 accuracy just guessing 0, relatively balanced
compas_df["two_year_recid"].value_counts()

# check for any big differences in average recidivism by feature
# there are pretty substantial differences in all crime counts
# including juvenile misdemeanors, felones, other, and total prior counts
# there is also a 4 year gap showing younger people tend to reoffend, on average
average_recid = compas_df.groupby('two_year_recid').mean()
average_by_race = compas_df.groupby('race').mean()

# check pairplot for any visual cues on separation by recidivism
sns.pairplot(compas_df, hue='two_year_recid');

# check for any oddities in distribution of features
compas_df.describe()

# there appears to be a jump in reoffending for anyone committing at least 1 juvenile crime
# create boolean features for both juvenile misdemeanors and felonies
compas_df["juv_m_gt1"] = compas_df["juv_misd_count"].apply(lambda x: 0 if x == 0 else 1)
compas_df["juv_f_gt1"] = compas_df["juv_fel_count"].apply(lambda x: 0 if x == 0 else 1)
compas_df["felony"] = compas_df["c_charge_degree"].apply(lambda x: 0 if x == "M" else 1)
compas_df["female"] = compas_df["sex"].apply(lambda x: 0 if x == "Male" else 1)

# plots of potential features created to look for differences in recidivism likelihood
pd.crosstab(compas_df.juv_m_gt1, compas_df.two_year_recid.astype(bool)).plot(kind='bar')
pd.crosstab(compas_df.juv_f_gt1, compas_df.two_year_recid.astype(bool)).plot(kind='bar')
pd.crosstab(compas_df.felony, compas_df.two_year_recid.astype(bool)).plot(kind='bar')
pd.crosstab(compas_df.sex, compas_df.two_year_recid.astype(bool)).plot(kind='bar')

# create boolean to test priors count by race
compas_df.race.value_counts()
compas_df["race_black"] = compas_df["race"].apply(lambda x: 1 if x == "African-American" else 0)
compas_df["race_white"] = compas_df["race"].apply(lambda x: 1 if x == "Caucasian" else 0)

white_df = compas_df[compas_df["race_white"] == 1]
black_df = compas_df[compas_df["race_black"] == 1]

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


    

    
    