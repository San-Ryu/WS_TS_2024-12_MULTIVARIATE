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
## 단일 년도에 대한 국경일 정보 조회
## Input
##  1) year_tar : 대상년도
##  2) str_key : 발급받은 개인 키
## Output
##  1) 해당 휴일 정보가 포함된 데이터셋
def KASI_holiDay(year_tar, str_key):
    ## Parameter for Request
    url = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getHoliDeInfo"
    key = str_key

    params = "?" + urlencode({
        quote_plus("ServiceKey") : key
        , quote_plus("_type") : "json"
        , quote_plus("solYear") : str(year_tar)
        , quote_plus("numOfRows") : 100
    })

    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=600).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["response"]["body"]["items"]["item"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')

## 단일 년도에 대한 공휴일 정보 조회
## Input
##  1) year_tar : 대상년도
##  2) str_key : 발급받은 개인 키
## Output
##  1) 해당 휴일 정보가 포함된 데이터셋
def KASI_restDay(year_tar, str_key):
    ## Parameter for Request
    url = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
    key = str_key

    params = "?" + urlencode({
        quote_plus("ServiceKey") : key
        , quote_plus("_type") : "json"
        , quote_plus("solYear") : str(year_tar)
        , quote_plus("numOfRows") : 100
    })

    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=600).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["response"]["body"]["items"]["item"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')

## 단일 년도에 대한 기념일 정보 조회
def KASI_anniDay(year_tar, str_key):
    ## Parameter for Request
    url = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getAnniversaryInfo"
    key = str_key

    params = "?" + urlencode({
        quote_plus("ServiceKey") : key
        , quote_plus("_type") : "json"
        , quote_plus("solYear") : str(year_tar)
        , quote_plus("numOfRows") : 100
    })

    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=600).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["response"]["body"]["items"]["item"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')
#endregion Crawling_OpenData : DATE