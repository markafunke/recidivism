# Debiasing Recidivism Classification Models
A classification analysis of 2-year recidivism in Broward County, FL. 

## Background and Goals

ProPublica conducted [an analysis](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) on an algorithm devloped by Northpointe (COMPAS) which is used to predict the risk of recidivism in criminals. They found that it's biased against Black people, disproportionately misclassifying Black people as "high risk" that ultimately did not reoffend, and White people as "low risk" that did reoffend. ProPublica made this data publicly available on their github.

The goal of this analysis is to replicate ProPublica's findings, and create a model of my own that is not racially biased.

## Conclusions 

Leveraging sci-kit lego's [InformationFilter](https://scikit-lego.readthedocs.io/en/latest/fairness.html), I filtered information from the "Race" column away from all remaining features as a transformation step in preprocessing. This ultimately led to significantly more equitable predictions across White and Black defendants from a logistic regression model. However, there is a slight tradeoff in overall accuracy. Read about my process and detailed conclusions in my blog post [HERE](https://medium.com/@markafunke/de-biasing-an-unjust-criminal-predictive-model-9f2fff4852e3).

## Outline of Files

#### Data Acquisition

*Note: If replicating this analysis, you do not need to run this step, you can either access the csv file directly, or start in the eda_preprocessing.py file that references the "compas_clean.p" pickle file that is the cleaned DataFrame resulting from this SQL code.*

- **create_tables.sql**: I ran this SQL code within DBeaver to store the [Compas Two Year Recidivism](https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv) data on an AWS server. This also limits the data in the same manner as the ProPublica analysis, resulting in 6172 observations.

#### Exploratory Data Analysis / Preprocessing

- **eda_preprocessing.py**: This code includes the feature engineering and intuition behind the features ultimately tested and used in the modeling process.

#### Modeling

- **classification_util.py**: Contains three functions used to score classification models in "modeling.py"
- **modeling.py**: Fits and evaluates recidivism classification models both pre and post transforming the data for fairness. Compares predictions for White and Black defendants to evaluate racial bias, and exports the final predictions for use in a Tableau demonstration.
- **hyperparameter_tuning.py**: This file was used in the iterative process of fitting the best classification model. It uses GridSearchCV to fit the optimal parameters for each model tested based on the F(1/3) metric. These parameters are ultimately used in the "modeling.py" file. Since the parameters are already reflected in the "modeling.py "file, if re-creating this analysis, there is no need to run this file.
