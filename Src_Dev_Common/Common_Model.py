#region Info
## Hist
## [2024-04-05] Created

## Desc
## Dataset 처리에 있어서 공통으로 사용하는 기능들을 함수화
#endregion Info

#region Module_Import
## Basic Import
## Basic
import os
os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd
from pandas import DataFrame, Series

import math, random

## Datetime
import time
import datetime as dt
from datetime import datetime, date, timedelta

import glob
from glob import glob
import requests
import json

## 시각화
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = [10, 8]

from sklearn.model_selection import train_test_split, KFold, GridSearchCV

## Models
## Catboost
import catboost as cb
from catboost import Pool, CatBoostRegressor

## LightGBM
import lightgbm as lgbm
from lightgbm import LGBMRegressor

## XGBoost
import xgboost as xgb
from xgboost import plot_importance, plot_tree, XGBClassifier

## Decision Tree
from sklearn.tree import DecisionTreeRegressor
## RandomForest
from sklearn.ensemble import RandomForestRegressor

## sklearn.metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_log_error, r2_score
#endregion Module_Import

#region List
## 시각화
# def resample_by_1H_last(df_tar, col_tar):
#endregion List

#region Model
## Data Split
## Input 
##  1) df_tar : 타겟 데이터 (Train/Test 분리 X 원본)
##  2) float_rate : Test 비율
##  3) str_col_tar : Target Column명
def data_train_test_split(df_tar, float_rate, str_col_tar):
    trainSet_Origin, testSet_Origin = train_test_split(df_tar, test_size = float_rate, shuffle = False)

    trainSet, testSet = trainSet_Origin, testSet_Origin

    ## Input / Target Split
    trainXX, trainYY = trainSet.drop([str_col_tar],axis=1), trainSet[[str_col_tar]]
    testXX, testYY = testSet.drop([str_col_tar],axis=1), testSet[[str_col_tar]]

    trainXXindex, trainYYindex = trainXX.index, trainYY.index
    trainXXcolumns, trainYYcolumns = trainXX.columns, trainYY.columns

    testXXindex, testYYindex = testXX.index, testYY.index
    testXXcolumns, testYYcolumns = testXX.columns, testYY.columns

    d_trainXX, d_trainYY = pd.DataFrame(trainXX, index=trainXXindex, columns=trainXXcolumns), trainYY

    d_testXX, d_testYY = pd.DataFrame(testXX, index=testXXindex, columns=testXXcolumns), testYY

    return d_trainXX, d_trainYY, d_testXX, d_testYY

## Model Predict
## Input 
##  1) int_model : 모델 번호
##  2) df_tar : 타겟 데이터 (Train/Test 분리 X 원본)
def model_ml_predict(df_trainXX, df_trainYY, df_testXX, df_testYY, int_model):
    ## Train Start
    tm_start = time.time()

    ## Model Analysis
    ## CB
    if int_model == 0: 
        # model = CatBoostRegressor(iterations = 500, max_ctr_complexity = 4, random_seed = 0
        #                           , od_type = 'Iter', od_wait = 25, verbose = 50
        #                           , depth = 4).fit(trainXX, trainYY, cat_features = []
        #                                            ,eval_set=[(trainXX, trainYY)])
        model = CatBoostRegressor(iterations = 500, max_ctr_complexity = 4, random_seed = 0
                                , od_type = 'Iter', od_wait = 25, verbose = 50
                                , depth = 4).fit(df_trainXX, df_trainYY, cat_features = []
                                                ,eval_set=[(df_trainXX, df_trainYY)])

    ## DT
    elif int_model == 1:
        # model = DecisionTreeRegressor(max_depth = 8).fit(trainXX, trainYY)
        model = DecisionTreeRegressor(max_depth = 8).fit(df_trainXX, df_trainYY)

    ## LGBM
    elif int_model == 2:
        # model = LGBMRegressor(n_estimators=10000, learning_rate=0.01, verbose=500).fit(trainXX, trainYY, eval_metric='mae'
        #                                                                                , eval_set=[(trainXX, trainYY)])
        model = LGBMRegressor(n_estimators=10000, learning_rate=0.01, verbose=500).fit(df_trainXX, df_trainYY, eval_metric='mae'
                                                                                    , eval_set=[(df_trainXX, df_trainYY)])

    ## RF
    elif int_model == 3:
        # model = RandomForestRegressor(max_depth = 8, min_samples_leaf = 8, min_samples_split = 8
        #                               , n_estimators = 200).fit(trainXX, trainYY)
        model = RandomForestRegressor(max_depth = 8, min_samples_leaf = 8, min_samples_split = 8
                                    , n_estimators = 200).fit(df_trainXX, df_trainYY)

    ## XGB
    elif int_model == 4:
        model = xgb.XGBRegressor(n_estimators = 1000).fit(df_trainXX, df_trainYY, eval_set=[(df_testXX, df_testYY)]
                                                        , early_stopping_rounds = 50, verbose = False)

    ## Train Over
    tm_code = time.time() - tm_start

    ## Visualization
    # if int_model == 2: ## LGBM
    #     lgbm.plot_importance(model)
    #     plt.show()
    # elif int_model == 3: ## RF
    #     ftr_importances_values = model.feature_importances_
    #     ftr_importances = pd.Series(ftr_importances_values, index=df_trainXX.columns)
    #     ftr_top = ftr_importances.sort_values(ascending=False)[:20]
    #     plt.figure(figsize=(8, 6))
    #     sns.barplot(x=ftr_top, y=ftr_top.index)
    #     plt.show()
    # elif int_model == 4: ## XGBoost
    #     ## 주요 변수 판단
    #     plot_importance(model)

    model_pred = model.predict(df_testXX)
    model_preds = np.reshape(model_pred,(-1,1))

    ## K-Fold
    # d_actual = df_testYY
    
    ## Non-Fold
    d_actual = df_testYY.to_numpy()
    d_actual = np.reshape(d_actual,(-1,1))

    return d_actual, model_preds, tm_code

## 단일 Model Analysis
## Input 
##  1) int_model : 모델 번호
##  2) df_tar : 타겟 데이터 (Train/Test 분리 X 원본)
def model_ml_analysis_single(df_tar, int_model, float_rate, str_col_tar):
    ## Data Split
    d_trainXX, d_trainYY, d_testXX, d_testYY = data_train_test_split(df_tar, float_rate, str_col_tar)
    
    d_actual, model_preds, tm_code = model_ml_predict(d_trainXX, d_trainYY, d_testXX, d_testYY, int_model)

    return d_actual, model_preds, tm_code

## KFold Model Analysis
## Input 
##  1) int_model : 모델 번호
##  2) df_tar : 타겟 데이터 (Train/Test 분리 X 원본)
def model_ml_analysis_with_KFold(df_tar, int_model, float_rate, str_col_tar, int_fold):
    ## 초기 변수 생성
    list_kf_mae, list_kf_mape, list_kf_mse, list_kf_rmse, list_kf_msle, list_kf_mbe, list_kf_r2, list_kf_tm_code = [], [], [], [], [], [], []
    list_kf_scores = [] ## 최종 출력될 Score List

    ## K-Fold 객체 생성
    k_fold = KFold(n_splits = int_fold)

    ## K-Fold 연산 수행
    for df_train, df_test in k_fold.split(df_tar):
        ## Data Split
        trainXX, trainYY = df_train.drop([str_col_tar],axis=1), df_train[[str_col_tar]]
        testXX, testYY = df_test.drop([str_col_tar],axis=1), df_test[[str_col_tar]]

        trainXXindex, trainYYindex = trainXX.index, trainYY.index
        trainXXcolumns, trainYYcolumns = trainXX.columns, trainYY.columns

        testXXindex, testYYindex = testXX.index, testYY.index
        testXXcolumns, testYYcolumns = testXX.columns, testYY.columns

        d_trainXX, d_trainYY = pd.DataFrame(trainXX, index = trainXXindex, columns = trainXXcolumns), trainYY
        d_testXX, d_testYY = pd.DataFrame(testXX, index = testXXindex, columns = testXXcolumns), testYY

        ## Model Analysis
        d_actual, model_preds, tm_code = model_ml_predict(d_trainXX, d_trainYY, d_testXX, d_testYY, int_model, float_rate, str_col_tar)
        ## Score 산출
        list_scores = model_sk_metrics(d_actual, model_preds)

        ## 각 list_score별 수집
        list_kf_mae.append(list_scores[0])
        list_kf_mape.append(list_scores[1])
        list_kf_mse.append(list_scores[2])
        list_kf_rmse.append(list_scores[3])
        list_kf_msle.append(list_scores[4])
        list_kf_mbe.append(list_scores[5])
        list_kf_r2.append(list_scores[6])
        list_kf_tm_code.append(list_scores[7])
    
    score_kf_mae = sum(list_kf_mae) / len(list_kf_mae)
    score_kf_mape = sum(list_kf_mape) / len(list_kf_mape)
    score_kf_mse = sum(list_kf_mse) / len(list_kf_mse)
    score_kf_rmse = sum(list_kf_rmse) / len(list_kf_rmse)
    score_kf_msle = sum(list_kf_msle) / len(list_kf_msle)
    score_kf_mbe = sum(list_kf_mbe) / len(list_kf_mbe)
    score_kf_r2 = sum(list_kf_r2) / len(list_kf_r2)
    score_kf_tm_code = sum(list_kf_tm_code) / len(list_kf_tm_code)

    list_kf_scores.append(score_kf_mae)
    list_kf_scores.append(score_kf_mape)
    list_kf_scores.append(score_kf_mse)
    list_kf_scores.append(score_kf_rmse)
    list_kf_scores.append(score_kf_msle)
    list_kf_scores.append(score_kf_mbe)
    list_kf_scores.append(score_kf_r2)
    list_kf_scores.append(score_kf_tm_code)

    ## [추가예정] list_d_actual / list_model_pred는 시각화 표현하고 싶을 때 변수로서 사용
    return list_kf_scores


## Define Metric : MBE
## Input
##  1) d_actual : 대조군 (실제 데이터)
##  2) model_preds : 실험군 (예측 데이터)
## Output
##  1) Metric
def mean_bias_error(d_actual, model_preds):
    mbe_loss = np.sum(d_actual - model_preds)/d_actual.size

    return mbe_loss

## Metrics
## Input
##  1) d_actual : 대조군 (실제 데이터)
##  2) model_preds : 실험군 (예측 데이터)
## Output
##  1) Metrics
def model_sk_metrics(d_actual, model_preds):
    score_mae = round(mean_absolute_error(d_actual, model_preds), 4)
    score_mape = round(mean_absolute_percentage_error(d_actual, model_preds), 4)
    score_mse = round(mean_squared_error(d_actual, model_preds), 4)
    score_rmse = round(mean_squared_error(d_actual, model_preds, squared = False), 4)
    # score_msle = round(mean_squared_log_error(d_actual, model_preds), 4)
    score_mbe = round(mean_bias_error(d_actual, model_preds), 4)
    score_r2 = round(r2_score(d_actual, model_preds), 4)

    list_scores = [score_mae, score_mape
                   , score_mse, score_rmse
                #    , score_msle
                   , score_mbe
                   , score_r2]

    ## Mean_Absolute_Error
    print('MAE  : ', score_mae)
    ## Mean_Absolute_Percentage_Error
    print('MAPE : ', score_mape)
    ## Mean_Squared_Error
    print('MSE  : ', score_mse)
    ## Root_Mean_Squared_Error
    print('RMSE : ', score_rmse)
    ## Mean_Squared_Log_Error
    # print('MSLE : ', score_msle)
    ## Mean_Bias_Error
    print('MBE : ', score_mbe)
    ## R2_Score
    print('R2   : ', score_r2)

    ## 필요시 변수화하여 사용
    return list_scores

## 시각화
## Input
##  1) d_actual : 대조군 (실제 데이터)
##  2) model_preds : 실험군 (예측 데이터)
##  3) str_title : Title
## Output
##  1) 시각화 : 대조군과 실험군에 대한 시각화 비교
def model_visualization(d_actual, model_preds, str_title):
    plt.figure(figsize=(30,5))
    plt.plot(d_actual,color='red',label='True')
    plt.plot(model_preds,color='blue',label='Pred')
    plt.title(str_title, fontsize = 30)
    plt.legend(fontsize = 20)
    plt.show()
#endregion Model