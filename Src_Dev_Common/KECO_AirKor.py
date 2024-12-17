## dev-shryu

#region Info
## Hist
## [2023-05-25] Created
## [2023-05-26] 기상청/한국환경공단 수집 기능 분리
## [2023-05-31] 현재년도 전일까지의 데이터 수집 기능 정상화
## [2023-11-17] 개선작업 진행
##               1) 공통코드 사용
##               2) 코드 간략화 및 개선
## [2023-03-14] 개선작업 진행
##               1) 공통코드 추가
##                   - Rename_KMA_ASOS_CSVDOWN
##                   - Interpolate_KMA_ASOS

## Desc
## 대한민국기상청 종관기상관측정보 (ASOS)
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

import glob, requests, json

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

## Excel/CSV
import openpyxl, xlrd

import urllib
from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus
#endregion Module_Import

#region List
## 시각화
# def resample_by_1H_last(df_tar, col_tar):
#endregion List

#region Crawling_OpenData
########## 관측소 메타데이터 관련 부분은 사용되지 않음 (관측소 번호 직접 사용) ##########
## KECO AirKorea 대기정보 Raw Data의 Column명을 변경
## Desc
##  : Public Data_Column명을 변경하고, 사용할 컬럼만 지정하여 df_tar로 출력
## Input
##  1) df_tar
##     ['지역'
##      , '측정소명', '측정소코드'
##      , '측정일시'
##      , 'SO2', 'CO', 'O3', 'NO2', 'PM10']
## Output
##  1) df_tar
##     ['METER_DATE'
##      , 'REGION', 'CD_OBSERVATORY', 'NM_OBSERVATORY'
##      , 'SO2', 'CO', 'O3', 'NO2', 'PM10']]
def Rename_KECO_AirKor(df_tar):
    df_tar = df_tar.rename(columns = {'지역' : 'REGION'
                                      , '측정소명' : 'NM_OBSERVATORY', '측정소코드' : 'CD_OBSERVATORY'
                                      , '측정일시' : 'METER_DATE'
                                      , 'SO2' : 'SO2', 'CO' : 'CO', 'O3' : 'O3', 'NO2' : 'NO2', 'PM10' : 'PM10'})
    df_tar = df_tar[['METER_DATE'
                     , 'REGION', 'CD_OBSERVATORY', 'NM_OBSERVATORY'
                     , 'SO2', 'CO', 'O3', 'NO2', 'PM10']]
    return df_tar

## 종관기상관측데이터에 대한 기본 Interpolation
## Desc
##  : KIER Energy Project_Weather Data에 대한 Interpolation 수행
## Input
##  1) df_tar
##     ['METER_DATE'
##      , 'REGION', 'CD_OBSERVATORY', 'NM_OBSERVATORY'
##      , 'SO2', 'CO', 'O3', 'NO2', 'PM10']]
## Output
##  1) df_tar : Input과 동일
def Interpolate_KMA_ASOS(df_tar):
    ## Date 형식 지정
    df_tar['METER_DATE'] = pd.to_datetime(df_tar['METER_DATE'])
    ## Data 보간
    df_tar['SO2'] = df_tar['SO2'].interpolate()
    df_tar['CO'] = df_tar['CO'].interpolate()
    df_tar['O3'] = df_tar['O3'].interpolate()
    df_tar['NO2'] = df_tar['NO2'].interpolate()
    df_tar['PM10'] = df_tar['PM10'].interpolate()
    return df_tar
#endregion Crawling_OpenData : DATE