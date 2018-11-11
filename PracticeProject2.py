# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:17:34 2018

@author: jsc426
"""
#binary/categorical outcome

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import xgboost as xgb



# Load data
flowers = load_iris()
flower_data = pd.DataFrame(flowers['data'], columns = flowers['feature_names'])
flower_type = pd.DataFrame(flowers['target'])

flower_type = flower_type.rename(columns = {0:'Flower Type'})

# Create binary outcome
condition = flower_type['Flower Type'] == 1
Binary_flower = pd.Series(np.where(condition, 1, 0))

# Split data Binary
Xtrain, Xtest, Ytrain, Ytest = train_test_split(flower_data, Binary_flower)

# Binary Analysis, examine training data
Xtrain.isnull().sum(axis = 0) #no NAN
Ytrain.value_counts()
Xtrain.describe()
sns.pairplot(flower_data)
# No big outliers, but features are correlated

# Create standardized data (create standardized polynomial terms after initial run)
scaler = StandardScaler().fit(Xtrain)
Xtrain_stand = pd.DataFrame(scaler.transform(Xtrain))
Xtest_stand = pd.DataFrame(scaler.transform(Xtest))

# PCA picture to get a better idea of group separation
pca = PCA(0.95)
pca.fit(Xtrain_stand)
pca_X = pd.DataFrame(pca.transform(Xtrain_stand))
pca_X.rename(columns = {0:'PC1', 1:'PC2'}, inplace = True)
pca_X['Flower'] = Ytrain.reset_index(level = 0).drop('index', axis = 1)
sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'Flower', data = pca_X)

"""looks like some overlap, but not too bad. first try with logistic where
I can experiment with the polynomial data and regularization (L1 and L2) and then
impliment boosting methods"""

# Create Polynomials and interactions
poly_2 = PolynomialFeatures(2)
poly_3 = PolynomialFeatures(3)

Xtrain_poly2 = pd.DataFrame(poly_2.fit_transform(Xtrain_stand)).drop(0, axis = 1)
Xtrain_poly3 = pd.DataFrame(poly_3.fit_transform(Xtrain_stand)).drop(0, axis = 1)

Xtest_poly2 = pd.DataFrame(poly_2.fit_transform(Xtest_stand)).drop(0, axis = 1)
Xtest_poly3 = pd.DataFrame(poly_3.fit_transform(Xtest_stand)).drop(0, axis = 1)

# Data for modeling
"""All models will be run with the Training, Standardized Training, Polynomial/Interaction
of the 2nd order, and the Polynomial/Interaction of the 3rd order to determine which features
are best for each model type""""

#Xtrain
#Xtrain_stand
#Xtrain_poly2
#Xtrain_poly3
X_data = Xtrain
X_data_reg = Xtrain_poly2
X_data_test_reg = Xtest_poly2
X_data_test = Xtest

"""grid search will allow for tuning of parameters and cross validation on the 
training set. For the K-fold CV, both K = 5 and 10 will be implimented and parameters
comapred to assess bias-variance trade off (as K increases, bias descreases and variance
increases)"""

#Logistic
"""inclusion of so many features in logistic will lead to overfitting. Regularization using
L1 and L2 penalties can help to smooth the model/combat overfitting. L1 is able to zero out
features, which acts as model selection and returns a sparse model. L2 cannot zero out
features (all features are retained in the model), but is able to smooth them towards zero 
(strong relationship with PCA)"""

#Ridge - L2
params_logit_Ridge = {'penalty':['l2'],
          'C':np.array([0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5])}

logistic = LogisticRegression(solver = 'lbfgs')
grid_logit_L2_CV5 = GridSearchCV(logistic, params_logit_Ridge, cv = 5)
grid_logit_L2_CV5.fit(X_data_reg, Ytrain)

grid_logit_L2_CV10 = GridSearchCV(logistic, params_logit_Ridge, cv = 10)
grid_logit_L2_CV10.fit(X_data_reg, Ytrain)

print('CV 5 is:', grid_logit_L2_CV5.best_estimator_.C, 'CV 10 is:', grid_logit_L2_CV10.best_estimator_.C)

""" Xtrain: CV 5 sets C = 2.0 and CV 10 C = 3.5
    Xtrain Standardized: CV 5 C = 1.0 and CV 10 C = 3.0
    Poly 2 CV 5 C = 1.5 and CV 10 C = 1.0
"""

# CV 5
logistic_L2_CV5 = LogisticRegression(C = grid_logit_L2_CV5.best_estimator_.C)
logistic_L2_CV5.fit(X_data_reg, Ytrain)
logit_L2_CV5_pred = logistic_L2_CV5.predict(X_data_reg)

pd.crosstab(Ytrain, logit_L2_CV5_pred)
print(np.mean(logit_L2_CV5_pred == Ytrain))
""" Xtrain: 0.723 overall accuracy
    Xtrain Standardized: 0.75 overall accuracy
    Poly2 1.0 overall accuracy
"""

# CV 10
logistic_L2_CV10 = LogisticRegression(C = grid_logit_L2_CV10.best_estimator_.C)
logistic_L2_CV10.fit(X_data_reg, Ytrain)
logit_L2_CV10_pred = logistic_L2_CV10.predict(X_data_reg)

pd.crosstab(Ytrain, logit_L2_CV10_pred)
print(np.mean(logit_L2_CV10_pred == Ytrain))
""" Xtrian: 0.7321 overall accuracy
    Xtrain Standardized: 0.768 overall accuracy
    Poly2 1.0 overall accuracy
"""

#Lasso - L1
params_logit_Lasso = {'penalty':['l1'],
          'C':np.array([0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5])}

logistic = LogisticRegression(solver =  'liblinear')
grid_logit_L1_CV5 = GridSearchCV(logistic, params_logit_Lasso, cv = 5)
grid_logit_L1_CV5.fit(X_data_reg, Ytrain)

grid_logit_L1_CV10 = GridSearchCV(logistic, params_logit_Lasso, cv = 10)
grid_logit_L1_CV10.fit(X_data_reg, Ytrain)

print('CV 5 is:', grid_logit_L1_CV5.best_estimator_.C, 'CV 10 is:', grid_logit_L1_CV10.best_estimator_.C)

""" Xtrain: CV 5 sets C = 3.5 and CV 10 C = 2.5
    Xtrain Standardized CV 5 C = 5.0 and CV 10 C = 4.0
    Poly2 CV5 and CV10 set C = 0.5
"""

# CV 5
logistic_L1_CV5 = LogisticRegression(C = grid_logit_L1_CV5.best_estimator_.C, solver =  'liblinear')
logistic_L1_CV5.fit(X_data_reg, Ytrain)
logit_L1_CV5_pred = logistic_L1_CV5.predict(X_data_reg)

pd.crosstab(Ytrain, logit_L1_CV5_pred)
print(np.mean(logit_L1_CV5_pred == Ytrain))
""" Xtrain: 0.723 overall accuracy
    Xtrain Standardized 0.768 overall accuracy
    Poly2 1.0 overall accuracy
 """

# CV 10
logistic_L1_CV10 = LogisticRegression(C = grid_logit_L1_CV10.best_estimator_.C, solver =  'liblinear')
logistic_L1_CV10.fit(X_data_reg, Ytrain)
logit_L1_CV10_pred = logistic_L1_CV10.predict(X_data_reg)

pd.crosstab(Ytrain, logit_L2_CV10_pred)
print(np.mean(logit_L2_CV10_pred == Ytrain))
""" Xtrain: 0.7321 overall accuracy
    Xtrain Standardized 0.768 overall accuracy
    Poly2 1.0 overall accuracy
"""

"""Clearly the Poly2 helps capture enough of the non-linearity in the data
that there is perfect classification (possible overfitting). No need to run Poly3."""

# RF
"""Random Forests are useful in situations where there is non-linearity
and outliers. The algorithm partitions up the feature space and makes predictions in each
partition. This makes it so outliers don't influece the predictions in other areas of the 
feature space. This kind of algorithm is essentially a step function and does well with non-linearity
and high-order interactions among the features, but doesn't do as well when the underlying
trend/relationship is linear

** additionally for RF and the following boosting methods, scale of the variables and handling of 
categorical variables are naturally handled through tree methods...so no need for standardizing**""""


params_RF = {'n_estimators':np.array([25, 50, 100, 250, 500]),
            'criterion':['gini', 'entropy'],
            'max_depth':np.array([5, 10, 15, 20, 25])}
            
RF = RandomForestClassifier()
grid_RF_CV5 = GridSearchCV(RF, params_RF, cv = 5)
grid_RF_CV5.fit(X_data, Ytrain)
print(grid_RF_CV5.best_estimator_)

grid_RF_CV10 = GridSearchCV(RF, params_RF, cv = 10)
grid_RF_CV10.fit(X_data, Ytrain)
print(grid_RF_CV10.best_estimator_)

"""CV 5: n_estimators = 25, criterion = entropy, and max_depth = 25 
   CV 10:  n_estimators = 25, criterion = gini, and max_depth = 5
"""

# CV 5
RF_CV5 = RandomForestClassifier(n_estimators = grid_RF_CV5.best_estimator_.n_estimators, 
                            criterion = grid_RF_CV5.best_estimator_.criterion,
                            max_depth = grid_RF_CV5.best_estimator_.max_depth)
RF_CV5.fit(X_data, Ytrain)
RF_CV5_pred = RF_CV5.predict(X_data)

pd.crosstab(Ytrain, RF_CV5_pred)
print(np.mean(RF_CV5_pred == Ytrain))

# CV 10
RF_CV10 = RandomForestClassifier(n_estimators = grid_RF_CV10.best_estimator_.n_estimators, 
                            criterion = grid_RF_CV10.best_estimator_.criterion,
                            max_depth = grid_RF_CV10.best_estimator_.max_depth)
RF_CV10.fit(X_data, Ytrain)
RF_CV10_pred = RF_CV10.predict(X_data)

pd.crosstab(Ytrain, RF_CV10_pred)
print(np.mean(RF_CV10_pred == Ytrain))

"""Xtrain was 1.0 for both CV 5 and 10 """"

#ADABoost
"""Adaptive Boosting works well in situations where there is non-linearity and
when one group is significantly larger than the other. The algorithm makes a prediction using
a weak learner, calculates a weighted error based on misclassifications, uses that error to
weight how much that iterations contributes to the overall prediction (voting power),
and finally reweights all observations based on if they were misclassified or not 
(misclassified points have their weight increased). Each iteration 'votes' and the 
cummulative vote determines the prediction (vote weight depends on accuracy of iteration's model).
This is why unbalanced classification problems often benefit from Adaptive Boosting,
but makes the algorithm vulnerable to outliers"""

params_ADA = {'n_estimators':np.array([25, 50, 100, 250, 500]),
              'learning_rate':np.array([0.25, 0.5, 1, 1.5, 2])}

ADA = AdaBoostClassifier()
grid_ADA_CV5 = GridSearchCV(ADA, params_ADA, cv = 5)
grid_ADA_CV5.fit(X_data, Ytrain)
print(grid_ADA_CV5.best_estimator_)

grid_ADA_CV10 = GridSearchCV(ADA, params_ADA, cv = 10)
grid_ADA_CV10.fit(X_data, Ytrain)
print(grid_ADA_CV10.best_estimator_)

""" CV 5: n_estimators = 25, learning_rate = 0.25
    CV 10: n_estimators = 25, learning_rate = 0.25
"""


ADA_CV5 = AdaBoostClassifier(n_estimators = grid_ADA_CV5.best_estimator_.n_estimators,
                         learning_rate = grid_ADA_CV5.best_estimator_.learning_rate)
ADA_CV5.fit(X_data, Ytrain)
ADA_pred_CV5 = ADA_CV5.predict(X_data)
pd.crosstab(Ytrain, ADA_pred_CV5)
print(np.mean(ADA_pred_CV5 == Ytrain))

""" Both had the same tuning parameters and 1.0 overall accuracy""""

#Gradient Boosting
"""Gradient Boosting works well in situations with non-linearity, but is susceptible to overfitting and outliers.
It trains a model to the data, uses the negative derivative of a loss fuction to calculate residuals(*), 
and then fits a model to the residuals(**). It iterates on (*) and (**), 
adding each model to the cummulative model using the learning rate
to determine how much influence that iteration has on the overall model. XGBoost, 
is useful because it is fast (C++) but it allows for L1 and L2 regularization, which helps
to decrease the overfitting of the model"""

#L2
params_GBM = {'n_estimators':np.array([25, 50, 100, 250, 500, 1000]),
              'learning_rate':np.array([0.01, 0.1, 0.5, 1]),
                'max_depth':np.array([2, 3, 5, 10]),
                'reg_lambda':np.array([0.5, 1, 1.5])}

GBM = xgb.XGBClassifier()
grid_GBM_CV5 = GridSearchCV(GBM, params_GBM, cv = 5)
grid_GBM_CV5.fit(X_data, Ytrain)
print(grid_GBM_CV5.best_estimator_)

grid_GBM_CV10 = GridSearchCV(GBM, params_GBM, cv = 10)
grid_GBM_CV10.fit(X_data, Ytrain)
print(grid_GBM_CV10.best_estimator_)

"""
    CV 5: n_estimators = 500, learning_rate = 0.01, max_depth = 2, reg_lambda = 0.5
    CV 10:n_estimators = 500, learning_rate = 0.01, max_depth = 2, reg_lambda = 0.5
"""

GBM = xgb.XGBClassifier(n_estimators = grid_GBM_CV5.best_estimator_.n_estimators,
                        learning_rate = grid_GBM_CV5.best_estimator_.learning_rate,
                        max_depth = grid_GBM_CV5.best_estimator_.max_depth,
                        reg_lambda = grid_GBM_CV5.best_estimator_.reg_lambda)
GBM.fit(X_data, Ytrain)
pred_GBM = GBM.predict(X_data)
pd.crosstab(Ytrain, pred_GBM)
print(np.mean(Ytrain == pred_GBM))

"""CV 5 and 10 tuned to the same values and have 1.0 overall accuracy """

#L1
params_GBM = {'n_estimators':np.array([25, 50, 100, 250, 500, 1000]),
              'learning_rate':np.array([0.01, 0.1, 0.5, 1]),
                'max_depth':np.array([2, 3, 5, 10]),
                'reg_alpha':np.array([0.5, 1, 1.5]),
                'reg_lambda':np.array([0])}

GBM = xgb.XGBClassifier()
grid_GBM_CV5 = GridSearchCV(GBM, params_GBM, cv = 5)
grid_GBM_CV5.fit(X_data, Ytrain)
print(grid_GBM_CV5.best_estimator_)

grid_GBM_CV10 = GridSearchCV(GBM, params_GBM, cv = 10)
grid_GBM_CV10.fit(X_data, Ytrain)
print(grid_GBM_CV10.best_estimator_)

"""
    CV 5: n_estimators = 500, learning_rate = 0.01, max_depth = 2, reg_alpha = 0.5
    CV 10:n_estimators = 500, learning_rate = 0.01, max_depth = 2, reg_alpha = 0.5
"""

GBM_L1 = xgb.XGBClassifier(n_estimators = grid_GBM_CV5.best_estimator_.n_estimators,
                        learning_rate = grid_GBM_CV5.best_estimator_.learning_rate,
                        max_depth = grid_GBM_CV5.best_estimator_.max_depth,
                        reg_alpha = grid_GBM_CV5.best_estimator_.reg_alpha,
                        reg_lambda = grid_GBM_CV5.best_estimator_.reg_lambda)
GBM_L1.fit(X_data, Ytrain)
pred_GBM = GBM_L1.predict(X_data)
pd.crosstab(Ytrain, pred_GBM)
print(np.mean(Ytrain == pred_GBM))

"""CV 5 and 10 tuned to the same values and have 1.0 overall accuracy """

# test stuff
logit_pred_test = logistic_L2_CV10.predict(Xtest_poly2)
pd.crosstab(Ytest, logit_pred_test)

RF_pred_test = RF_CV5.predict(X_data_test)
#RF_pred_test = RF_CV10.predict(X_data_test)
pd.crosstab(Ytest, RF_pred_test)

ADA_pred_test = ADA_CV5.predict(X_data_test)
pd.crosstab(Ytest, ADA_pred_test)

GBM_pred_test = GBM.predict(X_data_test)
pd.crosstab(Ytest, GBM_pred_test)

"""ADA and GB Boosting return the same confusion matrix:
col_0   0  1
row_0       
0      27  0
1       3  8

Logistic_L2_CV10:
col_0   0  1
row_0       
0      25  2
1       3  8

RF:
col_0   0  1
row_0       
0      25  2
1       2  9

the decision of what model to choose comes down to what is more dangerous:
a false positive or a false negative. The logistic model loses out on both
fronts, the boosting algos are best for avoiding false postives and
the RF better for false negatives...these are some serious flowers"""
