## dev-shryu

#region Info
## Hist
## [2023-11-29] Created
##               1) 공통코드 사용
##               2) 코드 간략화 및 개선

## Desc
## 한국지역난방공사_시간대별 열 공급량
## https://www.data.go.kr/data/15099319/fileData.do#/API%20%EB%AA%A9%EB%A1%9D/getuddi%3Aff86e691-7bf4-46b4-a828-e9ebda6aea1a
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

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

## 시각화
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = [10, 8]

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

#region Crawling_OpenData : DATE
## 시간대별 열공급량 OpenAPI
## Desc
##  : 
## Input
##  1) Observatory
##  2) str_key : "개인 Key 사용"
##  3) str_year
##  4) str_Interval : 아래와 같은 str에 따라 분기
##                     - "HR" : 시간별 
##                     - "DAY" : 일별
##  5) str_page_num : 페이지번호 
## Output
##  1) 
def KDHC_HEAT_Usage(str_key, str_page_num, str_ver  = "None"):
    ## Define Todate str
    # str_now_ymd = pd.datetime.now().date()
    # str_now_y = pd.datetime.now().year
    # str_now_m = pd.datetime.now().month
    # str_now_d = pd.datetime.now().day
    # str_now_hr = pd.datetime.now().hour
    # str_now_min = pd.datetime.now().minute

    if str_ver == "None" : str_ver = "v20220930"

    ## Parameter for Request
    url = "https://api.odcloud.kr/api/15099319/v1/"
    dict_version = {"v20181231" : "uddi:4ccf1119-648f-4b4a-b6f8-f66499741f25"
                    , "v20211231" : "uddi:87d90a27-4f90-4cf9-b0e8-bff7f352bfed"
                    , "v20220930" : "uddi:ff86e691-7bf4-46b4-a828-e9ebda6aea1a"}
    url = url + str(dict_version[str_ver])
    key = str_key

    params = "?" + urlencode({
        quote_plus("serviceKey") : key
        , quote_plus("page") : str_page_num
        , quote_plus("perPage") : "999" ## 페이지당 결과 수 (최대 1,000건)

        , quote_plus("totalCount") : 0
        , quote_plus("currentCount") : 0
        , quote_plus("matchCount") : 0
    })
    
    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=60).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["data"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')

## 종관기상관측데이터 OpenAPI Raw Data의 Column명을 변경
## Desc
##  : KIER Energy Project_Column명을 변경하고, 사용할 컬럼만 지정하여 df_tar로 출력
## Input
##  1) df_tar
##     ['tm', 'rnum', 'stnId', 'stnNm', 'ta', 'taQcflg', 'rn', 'rnQcflg', 'ws',
##      'wsQcflg', 'wd', 'wdQcflg', 'hm', 'hmQcflg', 'pv', 'td', 'pa',
##      'paQcflg', 'ps', 'psQcflg', 'ss', 'ssQcflg', 'icsr', 'dsnw', 'hr3Fhsc',
##      'dc10Tca', 'dc10LmcsCa', 'clfmAbbrCd', 'lcsCh', 'vs', 'gndSttCd',
##      'dmstMtphNo', 'ts', 'tsQcflg', 'm005Te', 'm01Te', 'm02Te', 'm03Te']
## Output
##  1) df_tar
##     ['METER_DATE'
##      , 'temp_outdoor', 'temp_dew_point', 'temp_ground'
##      , 'humidity'
##      , 'rainfall', 'snowfall', 'snowfall_3hr'
##      , 'wind_speed', 'wind_direction'
##      , 'pressure_vapor', 'pressure_area', 'pressure_sea'
##      , 'sunshine', 'solar_radiation'
##      , 'cloud_total', 'cloud_midlow'
##      , 'visual_range']
def KMA_ASOS_DATA(df_tar):
    Data_ASOS_tmp = Data_ASOS_tmp.rename(columns = {'tm' : 'METER_DATE'
                                                    , 'ta' : 'temp_outdoor'
                                                    , 'td' : 'temp_dew_point'
                                                    , 'ts' : 'temp_ground'
                                                    , 'hm' : 'humidity'
                                                    , 'rn' : 'rainfall'
                                                    , 'dsnw' : 'snowfall'
                                                    , 'hr3Fhsc' : 'snowfall_3hr'
                                                    , 'ws' : 'wind_speed'
                                                    , 'wd' : 'wind_direction'
                                                    , 'pv' : 'pressure_vapor'
                                                    , 'pa' : 'pressure_area'
                                                    , 'ps' : 'pressure_sea'
                                                    , 'ss' : 'sunshine'
                                                    , 'icsr' : 'solar_radiation'
                                                    , 'dc10Tca' : 'cloud_total'
                                                    , 'dc10LmcsCa' : 'cloud_midlow'
                                                    , 'vs' : 'visual_range'})

    Data_ASOS_tmp = Data_ASOS_tmp[['METER_DATE'
                                    , 'temp_outdoor', 'temp_dew_point', 'temp_ground'
                                    , 'humidity'
                                    , 'rainfall', 'snowfall', 'snowfall_3hr'
                                    , 'wind_speed', 'wind_direction'
                                    , 'pressure_vapor', 'pressure_area', 'pressure_sea'
                                    , 'sunshine', 'solar_radiation'
                                    , 'cloud_total', 'cloud_midlow'
                                    , 'visual_range']]
    
    return df_tar
#endregion Crawling_OpenData : DATE