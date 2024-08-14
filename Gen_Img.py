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
import io
import torch
import torch.nn as nn
from torch.autograd import variable
import tushare as ts
import torchvision.datasets
import mplfinance as mpf



csvFile=open('E:\\test.csv','w',encoding='GB2312',newline="")
csvWriter=csv.writer(csvFile)

ts.set_token('4e7497b0629e1bf909b3efc087eadf39ef7002f9f4d57aea9a9303b0')
pro=ts.pro_api()
'''
获取商品交易所所有主力合约代码
'''
Future_Type_List_DF_CZCE=pro.fut_basic(exchange='CZCE', fut_type='2', fields='ts_code,symbol,name')
Future_Type_List_DF_DCE=pro.fut_basic(exchange='DCE', fut_type='2', fields='ts_code,symbol,name')
Future_Type_List_DF=pd.concat([Future_Type_List_DF_CZCE,Future_Type_List_DF_DCE])
'''
将主力合约和连续合约分开
'''
Future_Type_List_MA=Future_Type_List_DF.loc[Future_Type_List_DF['symbol'].str[-1:]!='L']
Future_Type_List_LX=Future_Type_List_DF.loc[Future_Type_List_DF['symbol'].str[-1:]=='L']

print(Future_Type_List_MA)
'''
'''
def GetDailyDataByTsCode(Pro,Ts_code,Start_Date,End_Date):
    df=Pro.fut_daily(ts_code=Ts_code,start_date=Start_Date,end_date=End_Date,fields=
                     'ts_code,trade_date,open,high,low,close,vol,oi')
    return df

def GetALLDataFrame(Pro,IniDF,Type_Df,Start_Date,End_Date):
    for ts_code in zip(Type_Df['ts_code']):
        df=GetDailyDataByTsCode(Pro,ts_code[0],Start_Date,End_Date)
        IniDF=pd.concat([IniDF,df])
    IniDF['open']=IniDF['open'].astype('float32')
    IniDF['close'] = IniDF['close'].astype('float32')
    IniDF['high'] = IniDF['high'].astype('float32')
    IniDF['low'] = IniDF['low'].astype('float32')
    IniDF['vol'] = IniDF['vol'].astype('float32')
    return IniDF

def PlotImage(DF,FILENAME):
    my_color = mpf.make_marketcolors(up='r',down='g',edge='inherit',wick='inherit',volume='b')
    my_style = mpf.make_mpf_style(marketcolors=my_color,figcolor='(0.82, 0.83, 0.85)')
    fig = mpf.figure(style=my_style, figsize=(2.56, 2.56), facecolor=(0.82, 0.83, 0.85))
    ax1 = fig.add_axes([-0.06, 0.20, 1.11, 0.80])#[-0.06, 0.20, 1.11, 0.80] [0, 0.20, 1.0, 0.80]
    # 添加第二、三张图表时，使用sharex关键字指明与ax1在x轴上对齐，且共用x轴
    ax2 = fig.add_axes([-0.06, 0.00, 1.11, 0.20], sharex=ax1)#[-0.06, 0.00, 1.11, 0.20] [0, 0.00, 1.0, 0.20
    ax2.xaxis.tick_bottom()
    ap = mpf.make_addplot(DF[['MA5', 'MA10', 'MA20']], ax=ax1)
    mpf.plot(DF, ax=ax1, addplot=ap,style=my_style, type='candle', volume=ax2)
    fig.savefig(FILENAME)
    #fig.show()

def PlotImageTest(DF):
    kwargs = dict(
        type='candle',
        mav=(7, 30, 60),
        volume=True,
        title='\nA_stock %s candle_line',
        ylabel='OHLC Candles',
        ylabel_lower='Shares\nTraded Volume',
        figratio=(15, 10),
        figscale=5)
    mc = mpf.make_marketcolors(
        up='red',
        down='green',
        edge='i',
        wick='i',
        volume='in',
        inherit=True)
    s = mpf.make_mpf_style(
        gridaxis='both',
        gridstyle='-.',
        y_on_right=False,
        marketcolors=mc)

    mpf.plot(DF,
             **kwargs,
             style=s,
             show_nontrading=False,
             savefig='E:\\test2.jpg')
    mpf.show()
    #mpf.plot(DF,style=s,type='candle',volume=True,mav=[5,10,20],savefig='E:\save.jpg')
def DF_Iint(df):
    df.insert(loc=len(df.columns), column='dateindex', value=df['trade_date'])#增加一列日期列
    x = df.copy()
    x.loc[:, 'trade_date'] = pd.to_datetime(x.loc[:, 'trade_date'])  # 将数据类型转换为日期类型,可以直接获取相应年份的数据
    #x = x.set_index('trade_date')
    df = x
    #df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df.rename(columns={'vol': 'volume'}, inplace=True)
    df = df.iloc[::-1]

    df.insert(loc=len(df.columns), column='MA5', value=df['close'].rolling(5).mean())
    df.insert(loc=len(df.columns), column='MA10', value=df['close'].rolling(10).mean())
    df.insert(loc=len(df.columns), column='MA20', value=df['close'].rolling(20).mean())
    return df

def Plot_Save_Image(df,filename):
    PlotImage(df, filename)

def Plot_Save_Image_List(df_tscode,df_data,batch_size,tday,savepath):
    for ts_code in zip(df_tscode['ts_code']):
        df=df_data.loc[df_data['ts_code']==ts_code[0]]
        df=DF_Iint(df)
        print('开始生成['+ts_code[0]+']的图像..........')
        if batch_size < len(df) and df['open'].isnull().any()==False and df['close'].isnull().any()==False and df['high'].isnull().any()==False and df['low'].isnull().any()==False and df['volume'].isnull().any()==False:
            for index in range(batch_size,len(df)+1):
                tempdf=df[index-batch_size:index]
                #print(datetime.datetime.strptime(tempdf.index[0],'Y%m%d'))
                #print(tempdf['dateindex'][0])
                Lable=Get_Lable(tempdf,batch_size,tday)
                print('正在生成['+ts_code[0]+']第'+str(index-batch_size+1)+'张图片......')
                PlotImage(tempdf,savepath+Lable+'\\'+ts_code[0]+str((index-batch_size+1))+'_'+tempdf['dateindex'][0]+'_'+tempdf['dateindex'][-1]+'_'+Lable+'.png')
                #print(index)
#
def Image_Lable_1(df,TDay):
    if (df[TDay:TDay+1]['close'].values<=df[TDay:TDay+1]['MA5'].values) and (df[TDay:TDay+1]['close'].values<=df[TDay:TDay+1]['MA10'].values) and (df[TDay:TDay + 1]['close'].values <= df[TDay:TDay + 1]['MA20'].values) and (df[-1:]['close'].values >= df[-1:]['MA5'].values) and (df[-1:]['close'].values >= df[-1:]['MA10'].values) and (df[-1:]['close'].values >= df[-1:]['MA20'].values):
        return True
    else:
        return False

def Image_Lable_2(df,TDay):
    if (df[TDay:TDay+1]['close'].values>=df[TDay:TDay+1]['MA5'].values) and (df[TDay:TDay+1]['close'].values>=df[TDay:TDay+1]['MA10'].values) and (df[TDay:TDay + 1]['close'].values >= df[TDay:TDay + 1]['MA20'].values) and (df[-1:]['close'].values <= df[-1:]['MA5'].values) and (df[-1:]['close'].values <= df[-1:]['MA10'].values) and (df[-1:]['close'].values <= df[-1:]['MA20'].values):
        return True
    else:
        return False
def Image_Lable_3(df,batch_size):
    index_MA5=batch_size-5
    index_MA10=batch_size-10
    index_MA20=batch_size-20
    if (df[-1:]['MA5'].values>=df[index_MA5:index_MA5+1]['MA5'].values) and (df[-1:]['MA10'].values>=df[index_MA10:index_MA10+1]['MA10'].values) and (df[-1:]['MA20'].values>=df[index_MA20:index_MA20+1]['MA20'].values) and (df[-1:]['MA5'].values>=df[-1:]['MA10'].values) and (df[-1:]['MA10'].values>=df[-1:]['MA20'].values):
        return True
    else:
        return False

def Image_Lable_4(df,batch_size):
    index_MA5 = batch_size - 5
    index_MA10 = batch_size - 10
    index_MA20 = batch_size - 20
    if (df[-1:]['MA5'].values <= df[index_MA5:index_MA5 + 1]['MA5'].values) and (df[-1:]['MA10'].values <= df[index_MA10:index_MA10 + 1]['MA10'].values) and (df[-1:]['MA20'].values <= df[index_MA20:index_MA20 + 1]['MA20'].values) and (df[-1:]['MA5'].values <= df[-1:]['MA10'].values) and (df[-1:]['MA10'].values <= df[-1:]['MA20'].values):
        return True
    else:
        return False

def Get_Lable(df,batch_size,tday):
    if Image_Lable_1(df,tday)==True:
        return 'T1'
    elif Image_Lable_2(df,tday)==True:
        return 'T2'
    elif Image_Lable_3(df,batch_size):
        return 'T3'
    elif Image_Lable_4(df,batch_size):
        return 'T4'
    else:
        return'T5'




def test(Pro,start_date,end_date):
    ts_code='NULL'
    df=GetDailyDataByTsCode(Pro,ts_code,start_date,end_date)
    return df

'''
设置输入参数
'''
start_date='20210916'
end_date='20220731'
batch_size=60#每次需要的数据天数，一般为60日
Tday=-5
Img_Save_Path='D:\\PythonProject\\CNN_Predit\\DATA\\IMG\\ResNet\\SMALL\\train\\'

inidf_MA=GetDailyDataByTsCode(pro,'NULL',start_date,end_date)
inidf_LX=GetDailyDataByTsCode(pro,'NULL',start_date,end_date)

'''
获取主力和连续合约交易数据
'''
Result_DF_MA=GetALLDataFrame(pro,inidf_MA,Future_Type_List_MA,start_date,end_date)
#Result_DF_LX=GetALLDataFrame(pro,inidf_LX,Future_Type_List_LX,start_date,end_date)
#testdf=Result_DF_MA.loc[Result_DF_MA['ts_code']=='BB.DCE']
#testdf=GetDailyDataByTsCode(pro,'P.DCE',start_date,end_date)
'''
if testdf['open'].isnull().any()==False and testdf['open'].isnull().any()==False:
    testdf['trade_date'] = pd.to_datetime(testdf['trade_date'])
    testdf.set_index('trade_date', inplace=True)
    testdf.rename(columns={'vol': 'volume'}, inplace=True)
    testdf = testdf.iloc[::-1]
    testdf.insert(loc=len(testdf.columns), column='MA5', value=testdf['close'].rolling(5).mean())
    testdf.insert(loc=len(testdf.columns), column='MA10', value=testdf['close'].rolling(10).mean())
    testdf.insert(loc=len(testdf.columns), column='MA20', value=testdf['close'].rolling(20).mean())
'''
Plot_Save_Image_List(Future_Type_List_MA,Result_DF_MA,batch_size,Tday,Img_Save_Path)
#Plot_Save_Image_List(Future_Type_List_MA.loc[Future_Type_List_MA['ts_code']=='SA.ZCE'],Result_DF_MA.loc[Result_DF_MA['ts_code']=='SA.ZCE'],30,Img_Save_Path)
#testdf.insert(loc=len(testdf.columns),column='MA60',value=testdf['close'].rolling(60).mean())
    #ImgFileName=Img_Save_Path+'test1.png'
#testdf=DF_Iint(testdf)
#print (testdf)
    #PlotImage(testdf[0:60],ImgFileName)
#PlotImageTest(testdf)


#print(df)



'''
df=pro.fut_daily(exchange='CZCE', ts_code='SA2209.ZCE',start_date='20220101',end_date='20220630')

df1=pro.fut_basic(exchange='CZCE', fut_type='2')
df2=pro.fut_mapping(ts_code='SAL.ZCE')
df3=pro.fut_wsr(symbol='FG',)
print(df3)
data1=pd.DataFrame(df3)
print(df)
data1.to_csv('E:\\test.csv',encoding='GB2312')
'''