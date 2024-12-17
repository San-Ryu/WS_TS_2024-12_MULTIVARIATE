## dev-shryu

## Hist
## [2024-02-14] Created

## Desc
## Import 목록 (복붙용)

######################################## ▶ Module Import ########################################

#region ▶ Basic
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

import math
import random

## Datetime
import time
import datetime as dt
from datetime import datetime, date, timedelta

#region ▶ Web (JSON / HTML)
import urllib
from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus

import glob
from glob import glob
import requests
import json

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

#region ▶ Visualization
## 시각화
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = [10, 8]

#region ▶ Scipy
from scipy import stats

#region ▶ Scikit Learn
# Data Split
from sklearn.model_selection import train_test_split

# Clustering (중첩 허용) - L1 norm, L2 norm (Hierarchical Based)
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage

# Clustering (중첩 비허용) - K-Means (Distance Based)
from sklearn.cluster import KMeans, MiniBatchKMeans

# Clustering (중첩 비허용) - DBSCAN (Density Based)
from sklearn.cluster import DBSCAN

# Clustering Performance Metrics
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, silhouette_score, rand_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix

## 정규화
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics

## Init.
pd.options.display.float_format = '{:.10f}'.format
#endregion Basic_Import

######################################## Module Import ◀ ########################################