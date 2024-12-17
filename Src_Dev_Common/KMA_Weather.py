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

#region Crawling_OpenData : DATE
## 관측소 메타데이터 OpenAPI
## Desc
##  : 해당 API는 행정/공공기관에서만 사용 가능
## Input
##  1) 
## Output
##  1) 
########## 관측소 메타데이터 관련 부분은 사용되지 않음 (관측소 번호 직접 사용) ##########
def NWS_ASOS_META_Observatory():
    ## Parameter for Request
    url = "http://apis.data.go.kr/1360000/WethrBasicInfoService/getWhbuoyObsStn"
    key = "KEY"

    params = "?" + urlencode({
        quote_plus("ServiceKey") : key
        , quote_plus("pageNo") : 1
        , quote_plus("numOfRows") : "999" ## 페이지당 결과 수 (최대 1,000건)
        , quote_plus("dataType") : "JSON"
    })

    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=60).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["response"]["body"]["items"]["item"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')
########## 관측소 메타데이터 관련 부분은 사용되지 않음 (관측소 번호 직접 사용) ##########

## 종관기상관측데이터 OpenAPI
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
def KMA_ASOS_DATA(Observatory, str_key, year, str_Interval, str_page_num):
    ## Define Todate str
    str_now_ymd = pd.datetime.now().date()
    str_now_y, str_now_m, str_now_d = pd.datetime.now().year, pd.datetime.now().month, pd.datetime.now().day
    str_now_hr, str_now_min = pd.datetime.now().hour, pd.datetime.now().minute

    ## Parameter for Request
    url = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    # url = "http://apis.data.go.kr/1360000/AsosHourlyInfoService"
    key = str_key

    ## 시간값
    if(str(year) == str(str_now_y)) : date_end_YMD = str(year) + str(str_now_m) + str(str_now_d)
    else : date_end_YMD = str(year)+"1231"

    params = "?" + urlencode({
        quote_plus("ServiceKey") : key
        , quote_plus("pageNo") : str_page_num
        , quote_plus("numOfRows") : "99" ## 페이지당 결과 수 (최대 1,000건)
        , quote_plus("dataType") : "JSON"
        , quote_plus("dataCd") : "ASOS"
        , quote_plus("dateCd") : str_Interval
        , quote_plus("startDt") : str(year)+"0101"
        , quote_plus("startHh") : "00"
        , quote_plus("endDt") : date_end_YMD
        , quote_plus("endHh") : "23"
        , quote_plus("stnIds") : Observatory
    })
    
    req = urllib.request.Request(url + unquote(params))
    print(url + unquote(params))
    # print(req)
    response_body = urlopen(req, timeout=60).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["response"]["body"]["items"]["item"]
    print(data_items)

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
##      , 'humidity', 'rainfall', 'snowfall', 'snowfall_3hr'
##      , 'wind_speed', 'wind_direction'
##      , 'pressure_vapor', 'pressure_area', 'pressure_sea'
##      , 'sunshine', 'solar_radiation'
##      , 'cloud_total', 'cloud_midlow'
##      , 'visual_range']
def Rename_KMA_ASOS_API(df_tar):
    df_tar = df_tar.rename(columns = {'tm' : 'METER_DATE'
                                      , 'ta' : 'temp_outdoor', 'td' : 'temp_dew_point', 'ts' : 'temp_ground'
                                      , 'hm' : 'humidity', 'rn' : 'rainfall', 'dsnw' : 'snowfall', 'hr3Fhsc' : 'snowfall_3hr'
                                      , 'ws' : 'wind_speed', 'wd' : 'wind_direction'
                                      , 'pv' : 'pressure_vapor', 'pa' : 'pressure_area', 'ps' : 'pressure_sea'
                                      , 'ss' : 'sunshine', 'icsr' : 'solar_radiation'
                                      , 'dc10Tca' : 'cloud_total', 'dc10LmcsCa' : 'cloud_midlow'
                                      , 'vs' : 'visual_range'})
    df_tar = df_tar[['METER_DATE'
                     , 'temp_outdoor', 'temp_dew_point', 'temp_ground'
                     , 'humidity'
                     , 'rainfall', 'snowfall', 'snowfall_3hr'
                     , 'wind_speed', 'wind_direction'
                     , 'pressure_vapor', 'pressure_area', 'pressure_sea'
                     , 'sunshine', 'solar_radiation'
                     , 'cloud_total', 'cloud_midlow'
                     , 'visual_range']]
    return df_tar

## 종관기상관측데이터 Raw Data(CSV로 다운)의 Column명을 변경
## Desc
##  : KIER Energy Project_Column명을 변경하고, 사용할 컬럼만 지정하여 df_tar로 출력
## Input
##  1) df_tar
##     ['일시', '기온(°C)', '이슬점온도(°C)', '지면온도(°C)'
##      , '습도(%)', '강수량(mm)', '적설(cm)', '3시간신적설(cm)'
##      , '풍속(m/s)', '풍향(16방위)'
##      , '증기압(hPa)', '현지기압(hPa)', '해면기압(hPa)'
##      , '일조(hr)', '일사(MJ/m2)'
##      , '전운량(10분위)', '중하층운량(10분위)'
##      , '시정(10m)']
## Output
##  1) df_tar
##     ['METER_DATE'
##      , 'temp_outdoor', 'temp_dew_point', 'temp_ground'
##      , 'humidity', 'rainfall', 'snowfall', 'snowfall_3hr'
##      , 'wind_speed', 'wind_direction'
##      , 'pressure_vapor', 'pressure_area', 'pressure_sea'
##      , 'sunshine', 'solar_radiation'
##      , 'cloud_total', 'cloud_midlow'
##      , 'visual_range']
def Rename_KMA_ASOS_CSVDOWN(df_tar):
    df_tar = df_tar.rename(columns = {'일시' : 'METER_DATE'
                                      , '기온(°C)' : 'temp_outdoor', '이슬점온도(°C)' : 'temp_dew_point', '지면온도(°C)' : 'temp_ground'
                                      , '습도(%)' : 'humidity', '강수량(mm)' : 'rainfall', '적설(cm)' : 'snowfall', '3시간신적설(cm)' : 'snowfall_3hr'
                                      , '풍속(m/s)' : 'wind_speed', '풍향(16방위)' : 'wind_direction'
                                      , '증기압(hPa)' : 'pressure_vapor', '현지기압(hPa)' : 'pressure_area', '해면기압(hPa)' : 'pressure_sea'
                                      , '일조(hr)' : 'sunshine', '일사(MJ/m2)' : 'solar_radiation'
                                      , '전운량(10분위)' : 'cloud_total', '중하층운량(10분위)' : 'cloud_midlow'
                                      , '시정(10m)' : 'visual_range'})
    df_tar = df_tar[['METER_DATE'
                     , 'temp_outdoor', 'temp_dew_point', 'temp_ground'
                     , 'humidity'
                     , 'rainfall', 'snowfall', 'snowfall_3hr'
                     , 'wind_speed', 'wind_direction'
                     , 'pressure_vapor', 'pressure_area', 'pressure_sea'
                     , 'sunshine', 'solar_radiation'
                     , 'cloud_total', 'cloud_midlow'
                     , 'visual_range']]
    return df_tar

## 종관기상관측데이터에 대한 기본 Interpolation
## Desc
##  : KIER Energy Project_Weather Data에 대한 Interpolation 수행
## Input
##  1) df_tar
##     ['METER_DATE'
##      , 'temp_outdoor', 'temp_dew_point', 'temp_ground'
##      , 'humidity', 'rainfall', 'snowfall', 'snowfall_3hr'
##      , 'wind_speed', 'wind_direction'
##      , 'pressure_vapor', 'pressure_area', 'pressure_sea'
##      , 'sunshine', 'solar_radiation'
##      , 'cloud_total', 'cloud_midlow'
##      , 'visual_range']
## Output
##  1) df_tar : Input과 동일
def Interpolate_KMA_ASOS(df_tar):
    ## Date 형식 지정
    df_tar['METER_DATE'] = pd.to_datetime(df_tar['METER_DATE'])
    ## Data 보간
    df_tar['temp_outdoor'] = df_tar['temp_outdoor'].interpolate()
    df_tar['temp_dew_point'] = df_tar['temp_dew_point'].interpolate()
    df_tar['temp_ground'] = df_tar['temp_ground'].interpolate()
    df_tar['humidity'] = df_tar['humidity'].interpolate()
    df_tar['wind_speed'] = df_tar['wind_speed'].interpolate()
    df_tar['wind_direction'] = df_tar['wind_direction'].interpolate()
    df_tar['pressure_vapor'] = df_tar['pressure_vapor'].interpolate()
    df_tar['pressure_area'] = df_tar['pressure_area'].interpolate()
    df_tar['pressure_sea'] = df_tar['pressure_sea'].interpolate()
    ## Data 0으로 채움
    df_tar['rainfall'] = df_tar['rainfall'].fillna(0)
    df_tar['snowfall'] = df_tar['snowfall'].fillna(0)
    df_tar['snowfall_3hr'] = df_tar['snowfall_3hr'].fillna(0)
    df_tar['sunshine'] = df_tar['sunshine'].fillna(0)
    df_tar['solar_radiation'] = df_tar['solar_radiation'].fillna(0)
    df_tar['cloud_total'] = df_tar['cloud_total'].fillna(0)
    df_tar['cloud_midlow'] = df_tar['cloud_midlow'].fillna(0)
    df_tar['visual_range'] = df_tar['visual_range'].fillna(0)
    return df_tar
#endregion Crawling_OpenData : DATE