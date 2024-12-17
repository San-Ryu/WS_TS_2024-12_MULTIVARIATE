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
#endregion Module_Import

#region List
## 'DATE' 컬럼의 데이터 형식이 datetime으로서 유효한지 확인
# def validate_date(date_text, int_row, list_valueError):
      
## Invalid Date의 행 번호를 List로 저장
# def list_invalidDate(df_tar, col_tar):
      
## "Date" Column을 "년월일시분"으로 구분하여 저장
# def ord_time_data(df_tar, col_tar):
      
## 일정한 간격을 가진 시간 데이터셋 생성
# def create_df_dt_min(df_tar, int_dt_start, int_dt_end, str_interval):\
#endregion List

#region Datetime
## 'DATE' 컬럼의 데이터 형식이 datetime으로서 유효한지 확인
## Input
##  1) date_text : 검사할 DataCell
##  2) int_row : 검사할 Row (유효하지 않은 값의 Row값을 저장)
##  3) list_valueError : 유효하지 않은 값을 저장할 List
## Output
##  -
def validate_date(date_text, int_row, list_valueError):
	try:
		datetime.strptime(date_text,"%Y-%m-%d %H:%M:%S")
	except ValueError:
		# print("Incorrect data format({0}), should be %Y-%m-%d %H:%M:%S".format(date_text))
		# print(int_row)
		list_valueError.append(int_row)
		return False
	
## "validate_date" 함수를 사용,
## Invalid Date의 행 번호를 List로 저장
## Input
##  1) df_tar : 대상 Dataset
##  2) col_tar : 대상 Datatime Column
## Output
##  1) list_errValues : Error Row의 Index값 List
def list_invalidDate(df_tar, col_tar):
    print(df_tar.shape)
    list_errValues = [] 
    for i in range(0, len(df_tar)):
        validate_date(str(df_tar[col_tar].iloc[i]), i, list_errValues)
    print(len(list_errValues))
    print(list_errValues)

    return list_errValues

## 시간 데이터 구성 01
## "Date" Column을 "년월일시분"으로 구분하여 저장
## Input
##  1) df_tar : 대상 Dataset
##  2) col_tar : 대상 Datatime Column
## Output
##  1) df_tar : YMD Column이 추가된 Dataset
def create_col_ymdhm(df_tar, col_tar):
    ## 'DATE' 컬럼의 데이터 형식을 datetime으로 변경
    df_tar[col_tar] = pd.to_datetime(df_tar[col_tar])

    ## "DATE" 컬럼으로부터 YEAR, MONTH, DAY, HOUR, MINUTE, SECOND 생성
    df_tar['YEAR'] = df_tar[col_tar].dt.year
    df_tar['MONTH'] = df_tar[col_tar].dt.month
    df_tar['DAY'] = df_tar[col_tar].dt.day
    df_tar['HOUR'] = df_tar[col_tar].dt.hour
    df_tar['MINUTE'] = df_tar[col_tar].dt.minute
    # df_origin['SECOND'] = df_origin[col_tar].dt.second

    return df_tar

## 시간 데이터 구성 02
## "Datetime" Column을 통해 요일 데이터 구성
## Input
def create_col_weekdays(df_tar, tar_col):
    ## 요일코드 부여
    df_tar['day_of_the_week'] = df_tar[tar_col].dt.day_name()
    df_tar['code_day_of_the_week'] = 0
    dict_weekday ={"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4
                   , "Saturday":5, "Sunday":6}
    for i in range(0, len(df_tar)):
        df_tar['code_day_of_the_week'].iloc[i] = dict_weekday.get(df_tar['day_of_the_week'].iloc[i])

    return df_tar

## 시간 데이터 구성 03
## "YMDHM" 데이터를 통해 "Datetime" Column 구성
## Input
def create_col_datetime(df_tar, col_tar, col_y, col_m, col_d
                          , col_h = "None", col_min = "None", col_s = "None"):
    if col_h == "None" : df_tar[col_h] = 0
    if col_min == "None" : df_tar[col_min] = 0
    if col_s == "None" : df_tar[col_s] = 0

    df_tar[col_tar] = 0
    for i in range(0, len(df_tar)):
        df_tar[col_tar].iloc[i] = dt.datetime(int(df_tar[col_y].iloc[i])
                                              , int(df_tar[col_m].iloc[i])
                                              , int(df_tar[col_d].iloc[i])
                                              , int(df_tar[col_h].iloc[i])
                                              , int(df_tar[col_min].iloc[i])
                                              , int(df_tar[col_s].iloc[i]))
        
    return df_tar

## 시간 데이터 구성 04
## "Week of Year" Column을 통해 요일 데이터 구성
## Input
##  1) df_tar : 대상 Dataset
##  2) col_tar : 대상 Datatime Column
##  3) type_week : Week 출력 타입 ("Y-W" / "W" 등)
## Output
##  1) df_tar : YMD Column이 추가된 Dataset
def create_col_week(df_tar, tar_col, type_week):
    ## 요일코드 부여
    if type_week == "Y-W" : df_tar['WEEK'] = df_tar[tar_col].dt.strftime('%G-%V') 
    elif type_week == "W" : df_tar['WEEK'] = df_tar[tar_col].dt.strftime('%V') 

    return df_tar

## 일정한 간격을 가진 시간 데이터셋 생성
## Input
##  1) df_tar : 대상 Dataset
##  2) tar_col : 대상 Date Column
##  3) int_dt_start : 시점
##  4) int_dt_end : 종점
##  5) str_interval : 시간 간격 ("1min" / "1H" 등)
## Output
##  1) df_tar : 지정된 Period와 Interval을 가진 Dataframe
def create_df_dt(df_tar, tar_col, int_dt_start, int_dt_end, str_interval):
    ## 날짜 범위 생성 및 요일 부여
    df_tar[tar_col] = pd.date_range(start=str(int_dt_start)
                                 , end=str(int_dt_end)
                                 , freq=str_interval)
    df_tar['day_of_the_week'] = df_tar[tar_col].dt.day_name()

    ## 제거 : Rename은 밖에서 알아서 할 것
    # print(df_dateRange_h.info())
    # df_tar = df_tar.rename(columns={tar_col:tar_col})

    ## 년월일 분해
    df_tar = create_col_ymdhm(df_tar, tar_col)

    ## 요일코드 부여
    df_tar = create_col_weekdays(df_tar, tar_col)
    
    return df_tar

## 입력된 데이터셋의 Period를 출력
## Input
##  1) df_tar01 : 대상 Dataset 01
##  2) tar_col : 대상 Column
##  3) (Optional) df_tar02 : 대상 Dataset 02
## Output
##  1) Period의 시점 및 종점을 나타내는 두 datetime 변수
##      - date.min(), date.max()
##      - dt_date_Start, dt_date_End
def calc_df_dt(df_tar01, tar_col, df_tar02 = pd.DataFrame()):
      ## df_tar02가 주어지지 않은 경우
      if len(df_tar02) == 0:
        date = pd.to_datetime(df_tar01[tar_col])
        return date.min(), date.max()
      
      ## df_tar02가 주어진 경우
      elif len(df_tar02) == 0:
        date01 = pd.to_datetime(df_tar01[tar_col])
        date02 = pd.to_datetime(df_tar02[tar_col])

        ## 시점 도출을 위한 대소비교
        if date01.min() < date02.min() : dt_date_Start = date01.min()
        else : dt_date_Start = date02.min() ## date02의 최소값이 크거나, 둘 다 같거나
            
        ## 종점 도출을 위한 대소비교
        if date01.max() > date02.max() : dt_date_End = date01.max()
        else : dt_date_End = date02.max() ## date02의 최소값이 크거나, 둘 다 같거나
            

        return dt_date_Start, dt_date_End
## Optional Param은 사용하지 말자.
## https://yeonyeon.tistory.com/224

## 24시를 00시로 변환
## Ex) 2023-01-01 24:00   -->>   2023-01-02 00:00
## 문자열상 "24" 부분을 "00"으로 대체 후, timedelta를 이용하여 하루를 더함.
## --> ValueError: day is out of range for month
##     등과 같은 이월 상황에서의 Error 방지
## Ref
## https://gwoolab.tistory.com/34
def conv_midnight_24to00(df_tar, col_tar, col_src, _format):
    df_tar[col_tar] = 0
    for i in range(0, len(df_tar)):
        datetime = str(df_tar[col_src].iloc[i])

        try : df_tar[col_tar].iloc[i] = pd.to_datetime(datetime, format = _format)
        except:
            datetime = datetime[:-2] + '00'
            df_tar[col_tar].iloc[i] = pd.to_datetime(datetime, format = _format) + timedelta(days = 1)

    return df_tar
#endregion Datetime