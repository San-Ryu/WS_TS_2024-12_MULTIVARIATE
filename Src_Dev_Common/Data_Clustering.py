## dev-shryu

#region Info
## Hist
## [2024-04-03] Created  
## [2024-04-04] 공통코드화 개선 및 코드 간략화  
##              1) 한 함수 내에 K와 km이 함께 있는 경우 간략화
##                 (km에 이미 n_cluster가 포함되어있으므로)

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

# K-Means 알고리즘
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split

# CLustering 알고리즘의 성능 평가 측도
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, silhouette_score, rand_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix

#endregion Module_Import

#region Clustering
## ■ 단일회차 클러스터링
## CHI(Calinski Harabasz Index) 계수 산출
## Input
##  1) df_X_std : Dataframe (주로 정규화된 X로 입력)
##  2) int_cluster : CHI를 도출할 Cluster 수
## Output
##  1) calinski_harabasz_index : 계산된 calinski_harabasz_index
def get_calinski_harabasz_index(X, int_cluster):
    cluster_label = np.unique(int_cluster) ## 클러스터 라벨
    K = len(cluster_label) ## 총 클러스터 개수
    n = X.shape[0] ## 총 데이터 개수
    c = np.mean(X, axis=0) ## 전체 데이터 중심 벡터
    num_sum = 0 ## calinski_harabasz_index 분자
    denom_sum = 0 ## calinski_harabasz_index 분모
    for cl in cluster_label:
        sub_X = X[np.where(int_cluster == cl)[0], :]
        c_k = np.mean(sub_X, axis=0)
        n_k = sub_X.shape[0]
        num_sum += n_k*np.sum(np.square(c_k-c))
        denom_sum += np.sum(np.square(sub_X-c_k))
        
    ## calinski_harabasz_index
    calinski_harabasz_index = (num_sum/(K-1))/(denom_sum/(n-K))
    return calinski_harabasz_index

## 군집화에 따른 Cluster_Size 산출
## Input
##  1) km : 군집화 모델
##  2) df_X_std : Dataframe (주로 정규화된 X로 입력)
## Output
##  1) list_cnt_clusters : 군집화시 군집의 크기
def get_cluster_sizes(km, df_X_std):
    ## 초기 변수 생성
    K = km.n_clusters ## 군집의 수
    list_cnt_clusters = []

    ## k-means alogorithm 적합
    km.fit(df_X_std) 

    ## 결과: 레코드별 군집 라벨
    # print(km.labels_)
    for i in range(0, K):
        int_size_cluster = len(np.where(km.labels_ == i)[0])
        list_cnt_clusters.append(int_size_cluster)
        # print(int_size_cluster)

    # 결과: 군집별 컬럼별 중심평균
    # print(km.cluster_centers_)
    # 목적함수의 값
    # print(km.inertia_)

    return list_cnt_clusters

## 주어진 군집화 모델에 따른 군집화에 대한 시각화
## Input
##  1) str_interval : 데이터 간격
##  2) km : 군집화 모델
##  4) df_X_std : Dataframe (주로 정규화된 X로 입력)
## Output
##  1) 시각화 : 군집화 시각화
def clustering_visualization (str_interval, km, df_X_std):
    ## 초기 변수 생성
    K = km.n_clusters
    cluster = km.labels_

    color=['blue', 'green', 'orange', 'cyan', 'red', 'black', 'yellow', 'peru', 'purple', 'slategray']
    for k in range(K):
        data = df_X_std[cluster == k]
        plt.scatter(data[:, 0],data[:, 1], c=color[k], alpha=0.8, label='cluster %d' % k)
        plt.scatter(km.cluster_centers_[k, 0],km.cluster_centers_[k, 1], c='red', marker="x")
        plt.title('Clustering by ' + str_interval)
    plt.legend(fontsize=12, loc='upper right') # legend position
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

## 주어진 군집화 모델에 대한 평가지표 출력
## Input
##  1) str_interval : 데이터 간격
##  2) K : 군집의 수 (군집화 모델에 포함되어있으므로, 추후 삭제해야함)
##  3) km : 군집화 모델
##  4) df_X_std : Dataframe (주로 정규화된 X로 입력)
## Output
##  1) list_scores : 지표 List
##  2) print 출력 : 지표를 Print로 출력
def get_clustring_score (km, df_X_std, df_y):
    list_scores = []

    labels = km.labels_

    ## 실루엣 스코어
    score_sil = silhouette_score(df_X_std, labels, sample_size=1000)
    ## CHI 계수
    score_CHI = calinski_harabasz_score(df_X_std, labels)
    ## Davies_bouldin 스코어
    score_dbs = davies_bouldin_score(df_X_std, labels)

    ## Homogeneity 스코어
    score_homo = homogeneity_score(df_y, labels)
    ## Completeness 스코어
    score_comp = completeness_score(df_y, labels)
    ## V-Measure 스코어
    score_vMeasure = v_measure_score(df_y, labels)
    ## Rand-Index 스코어
    score_rand = rand_score(df_y, labels)
    ## Adjusted Rand-Index 스코어
    score_adjRand = adjusted_rand_score(df_y, labels)

    list_scores = [score_sil, score_CHI, score_dbs
                       , score_homo, score_comp, score_vMeasure, score_rand, score_adjRand]
    
    # print('contingency_matrix\n' , contingency_matrix(y, km.labels_))
    print('Silhouette Coefficient: %.4f' % score_sil)
    print('Calinski and Harabasz score: %.4f' % score_CHI)
    print("Davues Bouldin Score: %0.4f" % score_dbs)

    print("Homogeneity: %0.4f" % score_homo)
    print("Completeness: %0.4f" % score_comp)
    print("V-measure: %0.4f" % score_vMeasure)
    print("Rand-Index: %0.4f" % score_rand)
    print("Adjusted Rand-Index: %.4f" % score_adjRand) 
    
    ## 변수화하여 사용하고 싶을 경우, 해당 리스트에서 추출하여 사용 가능
    return list_scores



## ■ 다회차 클러스터링
## Elbow-method를 위한 Intertia 계산, 리스트 생성, 시각화
## Input
##  1) str_interval : 시각화 Graph에 출력될 데이터 간격
##  2) int_clusters_min : 최소 클러스터 수
##  3) int_clusters_max : 최대 클러스터 수
##  4) df_X_std : Dataframe (주로 정규화된 X로 입력)
## Output
##  1) list_intertia : 클러스터별 Intertia 수치
##  2) list_intertia_deriv : 클러스터별 Intertia차
##  3) 시각화 : Intertia Graph 시각화
def clustering_elbow_method (str_interval, int_clusters_min, int_clusters_max, df_X_std, opt_X = None):
    ## 예외처리01
    ## Min이 Max보다 크면 그냥 바꿔줌 + Int가 아니면 Int로 바꿔줌
    if int_clusters_min > int_clusters_max : int_clusters_min, int_clusters_max = int(int_clusters_max), int(int_clusters_min) + 1
    else : int_clusters_min, int_clusters_max = int(int_clusters_min), int(int_clusters_max) + 1
    
    ## 초기 변수  생성
    list_intertia, list_intertia_deriv = [], []
    K = range(int_clusters_min, int_clusters_max)

    ## K-Means
    for k in K:
        km = KMeans(n_clusters=k).fit(df_X_std)
        list_intertia.append(km.inertia_)
    
    ## list_intertia_deriv
    for i in range(1, len(list_intertia)):
        list_intertia_deriv.append(abs(float(list_intertia[i]) - float(list_intertia[i - 1])))

    ## Intertia Graph 시각화
    plt.plot(K, list_intertia, marker='.', markersize = 5, zorder = 2)
    plt.title('Inertia by number of clusters (Interval : ' + str_interval + ')')
    if opt_X != None : plt.scatter(opt_X, list_intertia[opt_X - 2], color = 'red', marker = '^', label = 'Point', zorder = 9999)
    plt.ylabel('Intertia')
    plt.xlabel('K')
    plt.show() 

    ## List를 출력하지 않고 변수로 Return
    return list_intertia, list_intertia_deriv


## Cluster 수에 따른 CHI계수 계산, 리스트 생성, 시각화
## Input
##  1) str_interval : 시각화 Graph에 출력될 데이터 간격
##  2) int_clusters_min : 최소 클러스터 수
##  3) int_clusters_max : 최대 클러스터 수
##  4) df_X_std : Dataframe (주로 정규화된 X로 입력)
## Output
##  1) list_Cluster_size : 해당 회차의 Cluster 크기
##  2) list_CHI : 해당 회차의 CHI
##  3) 시각화 : CHI Graph 시각화
def clustering_CHI_method (str_interval, int_clusters_min, int_clusters_max, df_X_std, opt_X = None):
    ## 예외처리01
    ## Min이 Max보다 크면 그냥 바꿔줌 + Int가 아니면 Int로 바꿔줌
    if int_clusters_min > int_clusters_max : int_clusters_min, int_clusters_max = int(int_clusters_max), int(int_clusters_min) + 1
    else : int_clusters_min, int_clusters_max = int(int_clusters_min), int(int_clusters_max) + 1

    ## 초기 변수  생성
    list_CHI = []
    K = range(int_clusters_min, int_clusters_max)

    for n_cluster in K:
        km_chi = KMeans(n_clusters = n_cluster
                        , init="k-means++"
                        , max_iter=300
                        , n_init=1).fit(df_X_std) 
        cluster = km_chi.predict(df_X_std)
        list_CHI.append(get_calinski_harabasz_index(df_X_std, cluster))

    fig = plt.figure(figsize=(8,8))
    fig.set_facecolor('white')
    ax = fig.add_subplot()
    ax.plot(K, list_CHI, marker='.', markersize = 5, zorder = 2)
    if opt_X != None : plt.scatter(opt_X, list_CHI[opt_X - 2], color = 'red', marker = '^', label = 'Point', zorder = 9999)
    ax.set_xticks(K)
    plt.xlabel('k')
    plt.ylabel('CHI')
    plt.title('Calinski-Harabasz Index by number of clusters (Interval : ' + str_interval + ')')
    plt.show()

    return list_CHI

## Cluster 수에 따른 Silhouette Score 계산, 리스트 생성, 시각화
## Input
##  1) str_interval : 시각화 Graph에 출력될 데이터 간격
##  2) int_clusters_min : 최소 클러스터 수
##  3) int_clusters_max : 최대 클러스터 수
##  4) df_X_std : Dataframe (주로 정규화된 X로 입력)
## Output
##  1) list_Silhouette : 해당 회차의 Silhouette 계수
##  2) list_cnt_clusters_by_K : 해당 회차의 Cluster 크기
##  3) 시각화 : Silhouette Score Graph 시각화
def clustering_Silhouette_method (str_interval, int_clusters_min, int_clusters_max, df_X_std, opt_X = None):
    ## 예외처리01
    ## Min이 Max보다 크면 그냥 바꿔줌 + Int가 아니면 Int로 바꿔줌
    if int_clusters_min > int_clusters_max : int_clusters_min, int_clusters_max = int(int_clusters_max), int(int_clusters_min) + 1
    else : int_clusters_min, int_clusters_max = int(int_clusters_min), int(int_clusters_max) + 1

    ## 초기 변수 생성
    list_Sil = [] ## List_Silhouette 계수
    list_cnt_clusters_by_K = [] ## List_군집의 크기
    K = range(int_clusters_min, int_clusters_max)

    for n_cluster in K:
        ## 초기 변수
        list_cnt_clusters = [] ## List : 군집의 수

        km_Sil = KMeans(n_clusters = n_cluster
                        , init="k-means++"
                        , max_iter=300
                        , n_init=1).fit(df_X_std)
        labels = km_Sil.labels_
        list_Sil.append(silhouette_score(df_X_std, labels, sample_size=1000))

        ## 결과: 레코드별 군집 라벨
        # print("■ " + str(n_cluster))
        for i in range(0, n_cluster):
            int_size_cluster = len(np.where(km_Sil.labels_ == i)[0])
            list_cnt_clusters.append(int_size_cluster)
            # print(int_size_cluster)

        list_cnt_clusters_by_K.append(list_cnt_clusters)

    fig = plt.figure(figsize=(8,8))
    fig.set_facecolor('white')
    ax = fig.add_subplot()
    ax.plot(K, list_Sil, marker='.', markersize = 5, zorder = 2)
    ax.set_xticks(K)
    if opt_X != None : plt.scatter(opt_X, list_Sil[opt_X - 2], color = 'red', marker = '^', label = 'Point', zorder = 9999)
    
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score by number of clusters (Interval : ' + str_interval + ')')
    plt.show()

    # plt.plot(K, list_Sil, 'bx-')
    # plt.title('Silhouette Score by number of clusters (Interval : ' + str_interval + ')')
    # plt.ylabel('Silhouette Score')
    # plt.xlabel('K')
    # plt.show() 

    return list_Sil, list_cnt_clusters_by_K

## 주어진 Time Interval에 따라 10회에 걸친 군집 형숭
## Input
##  1) K : 군집의 수
##  2) int_loop : 루프 반복 횟수
##  3) df_X_std : Dataframe (주로 정규화된 X로 입력)
## Output
##  1) list_log_clusters : int_loop 횟수에 걸친 군집화에 따른 군집의 크기
def clustering_get_cnt_by_loop (K, int_loop, df_X_std):
    ## 초기 변수 생성
    list_log_clusters = [] ## 군집의 수 총 기록

    for int_phase in range(0, int_loop):
        km = KMeans(n_clusters = K
            , init="k-means++"
            , max_iter=300
            , n_init=1)
        
        list_cnt_clusters = get_cluster_sizes(km, df_X_std)

        list_log_clusters.append(list_cnt_clusters)
        int_phase = int_phase + 1

    return list_log_clusters
#endregion Clustering
