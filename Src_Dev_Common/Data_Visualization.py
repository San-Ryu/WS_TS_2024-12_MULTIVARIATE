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

## 시각화
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = [10, 8]
#endregion Module_Import

#region List
## 시각화
# def resample_by_1H_last(df_tar, col_tar):
#endregion List

#region Visualization
## 시각화
## Input
##  1) df_tar : 대상 Dataset
##  2) col_tar : 대상 Datatime Column
## Output
##  -
def visualization_df(df_tar, str_tarCol, str_color):
    ## "METER_DATE" Column 생성
    df_tar['METER_DATE'] = 0
    for i in range(0, len(df_tar)):
        df_tar['METER_DATE'].iloc[i] = dt.datetime(int(df_tar['YEAR'].iloc[i])
                                                       , int(df_tar['MONTH'].iloc[i])
                                                       , int(df_tar['DAY'].iloc[i])
                                                       , int(df_tar['HOUR'].iloc[i])
                                                       , 0, 0)

    df_tar = df_tar[['METER_DATE', 'YEAR', 'MONTH', 'DAY'
                     , 'code_day_of_the_week'
                     , 'HOUR', str_tarCol]]

    ## 날짜 범위 지정
    date = pd.to_datetime(df_tar['METER_DATE'])

    ## 시각화
    fig, ax1 = plt.subplots(figsize=(30,5))
    title_font = {'fontsize': 20, 'fontweight': 'bold'}

    plt.title(str_tarCol, fontdict=title_font, loc='center', pad = 20)
    ax1.plot(date, df_tar[str_tarCol], color = str_color)
#endregion Visualization