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

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import random

## Datetime
import time
import datetime as dt
from datetime import datetime, date, timedelta

import glob
from glob import glob
import requests
import json

from Src_Dev_Common import Data_Analysis as com_Analysis
#endregion Module_Import

#region List
## resampling
# def resample_by_1H_last(df_tar, col_tar):

## IQR 방식으로 Outlier 제거
# def del_outlier_Usages(df_tar, col_tar):
#endregion List

#region Preprocessing
## resampling
## "Date" Column을 기준으로, 일정 간격으로 Resample (Last)
## Input
##  1) df_tar : 대상 Dataset
##  2) col_tar : 대상 Datatime Column
##  3) str_interval : 시간 간격 ('1min', '10min', '1H'...) 
## Output
##  1) df_tar : Resample된 Dataset
def resample_by_last(df_tar, str_domain, col_tar, str_interval):
    ## 확인차 다시 Datatime으로 형식 변경
    str_col_resample = str_domain + '_' + col_tar
    df_tar[str_col_resample] = pd.to_datetime(df_tar[col_tar])

    ## 
    df_tar = df_tar.dropna()
    df_tar = df_tar.resample(str_interval, on = str_col_resample).last()

    return df_tar

## IQR 방식으로 Outlier 제거
## 이상치 기준 생성 (IQR 방식)
## Q3 : 100개의 데이터로 가정 시, 25번째로 높은 값에 해당
## Q1 : 100개의 데이터로 가정 시, 75번째로 높은 값에 해당
## IQR : Q3 - Q1의 차이를 의미
## 이상치 : Q3 + 1.5 * IQR보다 높거나 Q1 - 1.5 * IQR보다 낮은 값을 의미
def del_outlier_Usages(df_tar, col_tar, int_cnt_process):
    q3_df_raw = df_tar[col_tar].quantile(0.90)
    q1_df_raw = df_tar[col_tar].quantile(0.25)    
    iqr_df_raw = q3_df_raw - q1_df_raw

    ## 이상치 갯수    
    cnt_outlier = 0

    for i in range(0, int_cnt_process):
        ## IQR 범위
        list_outlierRow = []
        list_outlierRow = com_Analysis.find_outlier_Usages(df_tar, col_tar)

        for row in list_outlierRow[::-1]:
            outlier_usage = df_tar[col_tar].iloc[row]
            if ((outlier_usage > (q3_df_raw + 1.5 * iqr_df_raw)) or (outlier_usage < q1_df_raw - 1.5 * iqr_df_raw)):
                # print(outlier_usage)
                ## Linear Regression
                df_tar[col_tar].iloc[row] = (df_tar[col_tar].iloc[row - 1] + df_tar[col_tar].iloc[row + 1]) / 2
                cnt_outlier = cnt_outlier + 1
            if outlier_usage < 0:
                # print(outlier_usage)
                ## 사용량이 음수일 수 없으므로 0으로 치환
                df_tar[col_tar].iloc[row] = np.nan
                cnt_outlier = cnt_outlier + 1
        
        i = i + 1

        if len(list_outlierRow) == 0:
            print(i)
            print("▶ ", str(cnt_outlier))
            list_outlierRow = com_Analysis.find_outlier_Usages(df_tar, col_tar)
            break

    print(cnt_outlier)
    
    return df_tar
#endregion Preprocessing