# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:11:08 2018

@author: soanand

Problem Assignment
I decided to treat this as a classification problem by creating a new binary variable affair
(did the woman have at least one affair?) and trying to predict the classification for each
woman.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
dta = sm.datasets.fair.load_pandas().data

# add "affair" column: 1 represents having affairs, 0 represents not
dta['affair'] = (dta.affairs > 0).astype(int)

y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + religious + educ + C(occupation) + C(occupation_husb)', dta, return_type="dataframe")

X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
						'C(occupation)[T.3.0]':'occ_3',
						'C(occupation)[T.4.0]':'occ_4',
						'C(occupation)[T.5.0]':'occ_5',
						'C(occupation)[T.6.0]':'occ_6',
						'C(occupation_husb)[T.2.0]':'occ_husb_2',
						'C(occupation_husb)[T.3.0]':'occ_husb_3',
						'C(occupation_husb)[T.4.0]':'occ_husb_4',
						'C(occupation_husb)[T.5.0]':'occ_husb_5',
						'C(occupation_husb)[T.6.0]':'occ_husb_6'})

y = np.ravel(y)

#split train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#create logistic model
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()

regressor.fit(X_train, y_train)

#predicting the result for test dataset 
y_pred = regressor.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)

 