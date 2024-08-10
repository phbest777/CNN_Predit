import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot
from pandas import to_datetime
import math,time
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import variable
import tushare as ts
import torchvision.datasets
import mplfinance as mplf
import cv2

csvFile=open('E:\\test.csv','w',encoding='GB2312',newline="")
csvWriter=csv.writer(csvFile)

ts.set_token('d0efdf1093648dfb0da4a3f99ae8db8878095a72bcc285246267718c')
pro=ts.pro_api()
df=pro.fut_daily(exchange='CZCE', ts_code='SA2209.ZCE',start_date='20220101',end_date='20220630')

df1=pro.fut_basic(exchange='CZCE', fut_type='2')
df2=pro.fut_mapping(ts_code='SAL.ZCE')
df3=pro.fut_wsr(symbol='FG',)
print(df3)
data1=pd.DataFrame(df3)
#print(df)
data1.to_csv('E:\\test.csv',encoding='GB2312')