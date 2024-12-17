#region Info
## Hist
## [2023-11-14] Created

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

## For OLS
from statsmodels.formula.api import ols

## sklearn.metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_log_error, r2_score
#endregion Module_Import

#region List
#endregion List

#region Analysis
## Descriptive Staticstic
## 기술통계량
## Input
##  1) date_text : 검사할 DataCell
##  2) int_row : 검사할 Row (유효하지 않은 값의 Row값을 저장)
##  3) list_valueError : 유효하지 않은 값을 저장할 List
## Output
##  -
def print_desc_statistic(df_tar, col_tar):
    ## Columns은 Domain별로 맞출 것
    print("=============== Descriptive Statistic ===============")
    print("Min of tar : " + str(np.min(df_tar[col_tar])))
    print("Std of tar : " + str(np.std(df_tar[col_tar])))
    print("Median of tar : " + str(np.median(df_tar[col_tar])))
    print("Mean of tar : " + str(np.mean(df_tar[col_tar])))
    print("Max of tar : " + str(np.max(df_tar[col_tar])))
    print("=============== Descriptive Statistic ===============")

    print("===============  IQR Range =============== ")
    list_outlierRows = []
    list_outlierRows = find_outlier_Usages(df_tar, col_tar)
    print("===============  IQR Range =============== ")



## IQR 방식으로 Outlier 탐지
## 이상치 기준 생성 (IQR 방식)
## Q3 : 100개의 데이터로 가정 시, 25번째로 높은 값에 해당
## Q1 : 100개의 데이터로 가정 시, 75번째로 높은 값에 해당
## IQR : Q3 - Q1의 차이를 의미
## 이상치 : Q3 + 1.5 * IQR보다 높거나 Q1 - 1.5 * IQR보다 낮은 값을 의미
def find_outlier_Usages(df_tar, col_tar):
    q3_df_raw = df_tar[col_tar].quantile(0.90)
    q1_df_raw = df_tar[col_tar].quantile(0.25)    
    iqr_df_raw = q3_df_raw - q1_df_raw
    
    ## IQR 범위
    print("===============  IQR Range =============== ")
    print(q3_df_raw + 1.5 * iqr_df_raw)
    print(q3_df_raw)
    print(iqr_df_raw)
    print(np.median(df_tar[col_tar]))
    print(q1_df_raw)
    print(q1_df_raw - 1.5 * iqr_df_raw)
    print("===============  IQR Range =============== ")

    ## 이상치 갯수    
    cnt_outlier = 0
    
    list_outlierRow = []
    for i in range(0, len(df_tar)):
        outlier_usage = df_tar[col_tar].iloc[i]
        if ((outlier_usage > (q3_df_raw + 1.5 * iqr_df_raw)) or (outlier_usage < q1_df_raw - 1.5 * iqr_df_raw)):
            # print(outlier_usage)
            list_outlierRow.append(i)            
            cnt_outlier = cnt_outlier + 1
        if outlier_usage < 0:
            # print(outlier_usage)
            list_outlierRow.append(i)
            cnt_outlier = cnt_outlier + 1
    print("cnt_outlier = " + str(cnt_outlier))
    
    return list_outlierRow



## Residual
## from statsmodels.formula.api import ols
def print_residual(df_tar):
    ## 잔차분석을 위한 Dataset 선언
    df_raw_residual = pd.DataFrame(df_tar)

    res = ols('temp_outdoor ~ HEAT_INST_561_1F', data = df_raw_residual).fit()
    print(res)

    # return resWW
#endregion Analysis