#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:28:38 2019

@author: gkalstn
"""
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

#csv파일 불러오기(천단위 ',' 빼고 읽기)
loan = pd.read_csv('/Users/gkalstn/Desktop/19-2/DM/prac/PBL/PBL/data.csv', thousands = ',')

#Label & feature 분리
X_loan = loan.drop(['OUTCOME'], axis = 1)

y_loan = loan['OUTCOME']

#데이터 분리(Train : 0.7, Test : 0.3) 
from sklearn.model_selection import train_test_split
Train_X, Test_X , Train_y , Test_y = train_test_split(X_loan, y_loan, test_size = 0.3, random_state = 1)

print("Train_X's shape:", Train_X.shape, ",\t Train_y's shape:", Train_y.shape)
print("Test_X's shape:", Test_X.shape, ",\t Test_y's shape:", Test_y.shape) # 150*0.3 = 45
print('\nhead of Train_X:\n', Train_X.head())


"""
======================================================
범주형 데이터 전처리 과정
First_home
Status
State
OUTCOME
"""

#First_home 전처리
Train_X['First_home'].value_counts()

Train_X.loc[Train_X["First_home"] == "Y", "First_home"] = 1
Train_X.loc[Train_X["First_home"] == "N", "First_home"] = 0


Test_X.loc[Test_X["First_home"] == "Y", "First_home"] = 1
Test_X.loc[Test_X["First_home"] == "N", "First_home"] = 0

#Status 전처리
#Pay_off == 1, Active == 0, Default == -1

Train_X['Status'].value_counts()

Train_X.loc[Train_X["Status"] == "Pay-off", "Status"] = 2
Train_X.loc[Train_X["Status"] == "Active", "Status"] = 1
Train_X.loc[Train_X["Status"] == "Default", "Status"] = 0


Test_X.loc[Test_X["Status"] == "Pay-off", "Status"] = 2
Test_X.loc[Test_X["Status"] == "Active", "Status"] = 1
Test_X.loc[Test_X["Status"] == "Default", "Status"] = 0

#State 집값낮은순으로 정렬
Train_X['State'].value_counts()

Train_X.loc[Train_X["State"] == "FL", "State"] = 52 - 19
Train_X.loc[Train_X["State"] == "CA", "State"] = 52 - 3
Train_X.loc[Train_X["State"] == "GA", "State"] = 52 - 27
Train_X.loc[Train_X["State"] == "TX", "State"] = 52 - 23
Train_X.loc[Train_X["State"] == "IL", "State"] = 52 - 32
Train_X.loc[Train_X["State"] == "NC", "State"] = 52 - 25
Train_X.loc[Train_X["State"] == "VA", "State"] = 52 - 16
Train_X.loc[Train_X["State"] == "NY", "State"] = 52 - 7
Train_X.loc[Train_X["State"] == "MD", "State"] = 52 - 11
Train_X.loc[Train_X["State"] == "AZ", "State"] = 52 - 21
Train_X.loc[Train_X["State"] == "PA", "State"] = 52 - 36
Train_X.loc[Train_X["State"] == "OH", "State"] = 52 - 49
Train_X.loc[Train_X["State"] == "MI", "State"] = 52 - 47
Train_X.loc[Train_X["State"] == "TN", "State"] = 52 - 28
Train_X.loc[Train_X["State"] == "NJ", "State"] = 52 - 10
Train_X.loc[Train_X["State"] == "WA", "State"] = 52 - 6
Train_X.loc[Train_X["State"] == "MO", "State"] = 52 - 44
Train_X.loc[Train_X["State"] == "NV", "State"] = 52 - 17
Train_X.loc[Train_X["State"] == "IN", "State"] = 52 - 43
Train_X.loc[Train_X["State"] == "OR", "State"] = 52 - 8
Train_X.loc[Train_X["State"] == "CO", "State"] = 52 - 5
Train_X.loc[Train_X["State"] == "WI", "State"] = 52 - 37
Train_X.loc[Train_X["State"] == "AL", "State"] = 52 - 38
Train_X.loc[Train_X["State"] == "SC", "State"] = 52 - 30
Train_X.loc[Train_X["State"] == "MN", "State"] = 52 - 46
Train_X.loc[Train_X["State"] == "MA", "State"] = 52 - 13
Train_X.loc[Train_X["State"] == "CT", "State"] = 52 - 12
Train_X.loc[Train_X["State"] == "LA", "State"] = 52 - 39
Train_X.loc[Train_X["State"] == "MS", "State"] = 52 - 4
Train_X.loc[Train_X["State"] == "UT", "State"] = 52 - 9
Train_X.loc[Train_X["State"] == "IA", "State"] = 52 - 48
Train_X.loc[Train_X["State"] == "KY", "State"] = 52 - 42
Train_X.loc[Train_X["State"] == "DE", "State"] = 52 - 20
Train_X.loc[Train_X["State"] == "ID", "State"] = 52 - 14
Train_X.loc[Train_X["State"] == "KS", "State"] = 52 - 41
Train_X.loc[Train_X["State"] == "AR", "State"] = 52 - 50
Train_X.loc[Train_X["State"] == "HI", "State"] = 52 - 1
Train_X.loc[Train_X["State"] == "NE", "State"] = 52 - 40
Train_X.loc[Train_X["State"] == "OK", "State"] = 52 - 45
Train_X.loc[Train_X["State"] == "NH", "State"] = 52 - 18
Train_X.loc[Train_X["State"] == "NM", "State"] = 52 - 33
Train_X.loc[Train_X["State"] == "RI", "State"] = 52 - 15
Train_X.loc[Train_X["State"] == "DC", "State"] = 52 - 2
Train_X.loc[Train_X["State"] == "ME", "State"] = 52 - 31
Train_X.loc[Train_X["State"] == "MT", "State"] = 52 - 24
Train_X.loc[Train_X["State"] == "WV", "State"] = 52 - 51
Train_X.loc[Train_X["State"] == "VT", "State"] = 52 - 26
Train_X.loc[Train_X["State"] == "ND", "State"] = 52 - 34
Train_X.loc[Train_X["State"] == "WY", "State"] = 52 - 29
Train_X.loc[Train_X["State"] == "AK", "State"] = 52 - 22
Train_X.loc[Train_X["State"] == "SD", "State"] = 52 - 35

Test_X.loc[Test_X["State"] == "FL", "State"] = 52 - 19
Test_X.loc[Test_X["State"] == "CA", "State"] = 52 - 3
Test_X.loc[Test_X["State"] == "GA", "State"] = 52 - 27
Test_X.loc[Test_X["State"] == "TX", "State"] = 52 - 23
Test_X.loc[Test_X["State"] == "IL", "State"] = 52 - 32
Test_X.loc[Test_X["State"] == "NC", "State"] = 52 - 25
Test_X.loc[Test_X["State"] == "VA", "State"] = 52 - 16
Test_X.loc[Test_X["State"] == "NY", "State"] = 52 - 7
Test_X.loc[Test_X["State"] == "MD", "State"] = 52 - 11
Test_X.loc[Test_X["State"] == "AZ", "State"] = 52 - 21
Test_X.loc[Test_X["State"] == "PA", "State"] = 52 - 36
Test_X.loc[Test_X["State"] == "OH", "State"] = 52 - 49
Test_X.loc[Test_X["State"] == "MI", "State"] = 52 - 47
Test_X.loc[Test_X["State"] == "TN", "State"] = 52 - 28
Test_X.loc[Test_X["State"] == "NJ", "State"] = 52 - 10
Test_X.loc[Test_X["State"] == "WA", "State"] = 52 - 6
Test_X.loc[Test_X["State"] == "MO", "State"] = 52 - 44
Test_X.loc[Test_X["State"] == "NV", "State"] = 52 - 17
Test_X.loc[Test_X["State"] == "IN", "State"] = 52 - 43
Test_X.loc[Test_X["State"] == "OR", "State"] = 52 - 8
Test_X.loc[Test_X["State"] == "CO", "State"] = 52 - 5
Test_X.loc[Test_X["State"] == "WI", "State"] = 52 - 37
Test_X.loc[Test_X["State"] == "AL", "State"] = 52 - 38
Test_X.loc[Test_X["State"] == "SC", "State"] = 52 - 30
Test_X.loc[Test_X["State"] == "MN", "State"] = 52 - 46
Test_X.loc[Test_X["State"] == "MA", "State"] = 52 - 13
Test_X.loc[Test_X["State"] == "CT", "State"] = 52 - 12
Test_X.loc[Test_X["State"] == "LA", "State"] = 52 - 39
Test_X.loc[Test_X["State"] == "MS", "State"] = 52 - 4
Test_X.loc[Test_X["State"] == "UT", "State"] = 52 - 9
Test_X.loc[Test_X["State"] == "IA", "State"] = 52 - 48
Test_X.loc[Test_X["State"] == "KY", "State"] = 52 - 42
Test_X.loc[Test_X["State"] == "DE", "State"] = 52 - 20
Test_X.loc[Test_X["State"] == "ID", "State"] = 52 - 14
Test_X.loc[Test_X["State"] == "KS", "State"] = 52 - 41
Test_X.loc[Test_X["State"] == "AR", "State"] = 52 - 50
Test_X.loc[Test_X["State"] == "HI", "State"] = 52 - 1
Test_X.loc[Test_X["State"] == "NE", "State"] = 52 - 40
Test_X.loc[Test_X["State"] == "OK", "State"] = 52 - 45
Test_X.loc[Test_X["State"] == "NH", "State"] = 52 - 18
Test_X.loc[Test_X["State"] == "NM", "State"] = 52 - 33
Test_X.loc[Test_X["State"] == "RI", "State"] = 52 - 15
Test_X.loc[Test_X["State"] == "DC", "State"] = 52 - 2
Test_X.loc[Test_X["State"] == "ME", "State"] = 52 - 31
Test_X.loc[Test_X["State"] == "MT", "State"] = 52 - 24
Test_X.loc[Test_X["State"] == "WV", "State"] = 52 - 51
Test_X.loc[Test_X["State"] == "VT", "State"] = 52 - 26
Test_X.loc[Test_X["State"] == "ND", "State"] = 52 - 34
Test_X.loc[Test_X["State"] == "WY", "State"] = 52 - 29
Test_X.loc[Test_X["State"] == "AK", "State"] = 52 - 22
Test_X.loc[Test_X["State"] == "SD", "State"] = 52 - 35

#Label 전처리

Train_y.value_counts()

Train_y.loc[Train_y == "non-default"] = 1
Train_y.loc[Train_y== "default"] = 0


Test_y.loc[Test_y == "non-default"] = 1
Test_y.loc[Test_y == "default"] = 0

#전처리 min-max 표준화
from scipy.stats import boxcox
#Train_X 전처리
Train_X['Bo_Age'] = (Train_X['Bo_Age'] - Train_X['Bo_Age'].min()) / (Train_X['Bo_Age'].max() - Train_X['Bo_Age'].min())
Train_X['Ln_Orig'] = (Train_X['Ln_Orig'] - Train_X['Ln_Orig'].min()) / (Train_X['Ln_Orig'].max() - Train_X['Ln_Orig'].min())
Train_X['Orig_LTV_Ratio_Pct'] = (Train_X['Orig_LTV_Ratio_Pct'] - Train_X['Orig_LTV_Ratio_Pct'].min()) / (Train_X['Orig_LTV_Ratio_Pct'].max() - Train_X['Orig_LTV_Ratio_Pct'].min())
Train_X['Credit_score'] = (Train_X['Credit_score'] - Train_X['Credit_score'].min()) / (Train_X['Credit_score'].max() - Train_X['Credit_score'].min())
Train_X['Tot_mthly_debt_exp'] = (Train_X['Tot_mthly_debt_exp'] - Train_X['Tot_mthly_debt_exp'].min()) / (Train_X['Tot_mthly_debt_exp'].max() - Train_X['Tot_mthly_debt_exp'].min())
Train_X['Tot_mthly_incm'] = (Train_X['Tot_mthly_incm'] - Train_X['Tot_mthly_incm'].min()) / (Train_X['Tot_mthly_incm'].max() - Train_X['Tot_mthly_incm'].min())
Train_X['orig_apprd_val_amt'] = (Train_X['orig_apprd_val_amt'] - Train_X['orig_apprd_val_amt'].min()) / (Train_X['orig_apprd_val_amt'].max() - Train_X['orig_apprd_val_amt'].min())
Train_X['pur_prc_amt'] = (Train_X['pur_prc_amt'] - Train_X['pur_prc_amt'].min()) / (Train_X['pur_prc_amt'].max() - Train_X['pur_prc_amt'].min())
Train_X['Status'] = (Train_X['Status'] - Train_X['Status'].min()) / (Train_X['Status'].max() - Train_X['Status'].min())
Train_X['Median_state_inc'] = (Train_X['Median_state_inc'] - Train_X['Median_state_inc'].min()) / (Train_X['Median_state_inc'].max() - Train_X['Median_state_inc'].min())
Train_X['State'] = (Train_X['State'] - Train_X['State'].min()) / (Train_X['State'].max() - Train_X['State'].min())

#Test_X 전처리
Test_X['Bo_Age'] = (Test_X['Bo_Age'] - Test_X['Bo_Age'].min()) / (Test_X['Bo_Age'].max() - Test_X['Bo_Age'].min())
Test_X['Ln_Orig'] = (Test_X['Ln_Orig'] - Test_X['Ln_Orig'].min()) / (Test_X['Ln_Orig'].max() - Test_X['Ln_Orig'].min())
Test_X['Orig_LTV_Ratio_Pct'] = (Test_X['Orig_LTV_Ratio_Pct'] - Test_X['Orig_LTV_Ratio_Pct'].min()) / (Test_X['Orig_LTV_Ratio_Pct'].max() - Test_X['Orig_LTV_Ratio_Pct'].min())
Test_X['Credit_score'] = (Test_X['Credit_score'] - Test_X['Credit_score'].min()) / (Test_X['Credit_score'].max() - Test_X['Credit_score'].min())
Test_X['Tot_mthly_debt_exp'] = (Test_X['Tot_mthly_debt_exp'] - Test_X['Tot_mthly_debt_exp'].min()) / (Test_X['Tot_mthly_debt_exp'].max() - Test_X['Tot_mthly_debt_exp'].min())
Test_X['Tot_mthly_incm'] = (Test_X['Tot_mthly_incm'] - Test_X['Tot_mthly_incm'].min()) / (Test_X['Tot_mthly_incm'].max() - Test_X['Tot_mthly_incm'].min())
Test_X['orig_apprd_val_amt'] = (Test_X['orig_apprd_val_amt'] - Test_X['orig_apprd_val_amt'].min()) / (Test_X['orig_apprd_val_amt'].max() - Test_X['orig_apprd_val_amt'].min())
Test_X['pur_prc_amt'] = (Test_X['pur_prc_amt'] - Test_X['pur_prc_amt'].min()) / (Test_X['pur_prc_amt'].max() - Test_X['pur_prc_amt'].min())
Test_X['Status'] = (Test_X['Status'] - Test_X['Status'].min()) / (Test_X['Status'].max() - Test_X['Status'].min())
Test_X['Median_state_inc'] = (Test_X['Median_state_inc'] - Test_X['Median_state_inc'].min()) / (Test_X['Median_state_inc'].max() - Test_X['Median_state_inc'].min())
Test_X['State'] = (Test_X['State'] - Test_X['State'].min()) / (Test_X['State'].max() - Test_X['State'].min())

cor_Train_X = Train_X.corr(method = 'pearson')

import matplotlib.pyplot as plt
plt.hist(Train_X['Status'])


#logistic
from sklearn.linear_model import LogisticRegression

log = LogisticRegression() #로지스틱 회귀분석 시행

log.fit(Train_X, Train_y) #모델의 정확도 확인
print('학습용 데이터셋 정확도 : %.2f' % log.score(Train_X, Train_y))
print('검증용 데이터셋 정확도 : %.2f' % log.score(Test_X, Test_y))

from sklearn.metrics import classification_report
y_pred=log.predict(Test_X)
print(classification_report(Test_y, y_pred))


