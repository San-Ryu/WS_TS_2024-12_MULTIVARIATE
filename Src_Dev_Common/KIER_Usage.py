## dev-shryu

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

#region Energy Data Treatment
## 

## 적산 데이터를 기반으로 순시 데이터를 생성
## Input
##  1) df_tar : 대상 Dataset
##  2) col_tar : 대상 적산값 Column
##  3) str_domain_tar : 대상 Domain명 (ELEC / HEAT / GAS / HOT / WATER)
## Output
##  1) 순시값 Column이 추가된 df_tar
def create_inst_usage (df_tar, col_tar, str_col_inst):
    df_tar[str_col_inst] = 0
    for i in range(0, len(df_tar) - 1) : df_tar[str_col_inst].iloc[i] = df_tar[col_tar].iloc[i + 1] - df_tar[col_tar].iloc[i]
    return df_tar
#endregion Energy Data Treatment