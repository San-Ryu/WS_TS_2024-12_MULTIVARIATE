## dev-shryu

#region Info
## Hist
## [2023-11-17] Created

## Desc
## 국경일/공휴일 정보 조회  
## 기념일은 제외, 근로자의 날은 수동으로 추가 예정)
##  --> 참고 : 근로자의 날은 2016년 1월 27일부로 법정 공휴일로 제정됨 
##      (https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EA%B7%BC%EB%A1%9C%EC%9E%90%EC%9D%98%EB%82%A0%EC%A0%9C%EC%A0%95%EC%97%90%EA%B4%80%ED%95%9C%EB%B2%95%EB%A5%A0)

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

#region Crawling_OpenData : Traffic
url = "https://www.bigdata-transportation.kr/api"

## [API] 톨게이트 목록 현황
## Input
##  1) str_key : 발급받은 개인 키
## Output
##  1) 톨게이트 목록
def KorEx_Tollgates(str_key):
    ## Parameter for Request
    # url = "https://www.bigdata-transportation.kr/api"
    key = str_key

    params = "?" + urlencode({
        quote_plus("apiKey") : key
        , quote_plus("productId") : "PRDTNUM_000000020307" ## 고유코드
        , quote_plus("numOfRows") : 999
    })

    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=600).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["result"]["unitLists"]#["items"]["item"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')

## [API] 영업소별 입구, 출구 현황
def KorEx_Tollgates_inOut(str_key, str_Tollgate, str_inOut = "None"):
    ## Parameter for Request
    # url = "https://www.bigdata-transportation.kr/api"
    key = str_key

    if str_inOut == "None":
        str_inOut = ""

    params = "?" + urlencode({
        quote_plus("apiKey") : key
        , quote_plus("productId") : "PRDTNUM_000000020305" ## 고유코드
        , quote_plus("unitCode") : str_Tollgate
        , quote_plus("inoutType") : str_inOut
    })

    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=600).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["result"]["laneStatusVO"]#["items"]["item"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')

## [Unused][API] 톨게이트 입/출구 교통량
## Input
##  1) str_key : 발급받은 개인 키
##  2) tmType : 자료 구분 ("1" : 1시간, "2" : 15분)
##  3) unitCode : 영업소 코드 (청주 : 111 / 남청주 : 112)
## Output
##  1) 톨게이트별 교통량 현황 (현재 시간 기준)
def KorEx_Tollgates_Traffic(str_key, tmType, unitCode):
    ## Parameter for Request
    # url = "https://www.bigdata-transportation.kr/api"
    key = str_key

    params = "?" + urlencode({
        quote_plus("apiKey") : key
        , quote_plus("productId") : "PRDTNUM_000000020308" ## 고유코드
        , quote_plus("tmType") : str(tmType) ## "1" : 1시간, "2" : 15분
        , quote_plus("unitCode") : str(unitCode) ## 청주 : 111 / 남청주 : 112
        , quote_plus("numOfRows") : 999
    })

    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=600).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["result"]["trafficIc"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')
#endregion Crawling_OpenData : DATE