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
import cx_Oracle
from sqlalchemy import create_engine
import pandas_oracle.tools as pt
#import cv2

'''
csvFile=open('E:\\test.csv','w',encoding='GB2312',newline="")
csvWriter=csv.writer(csvFile)

ts.set_token('4e7497b0629e1bf909b3efc087eadf39ef7002f9f4d57aea9a9303b0')
pro=ts.pro_api()
df=pro.fut_daily(exchange='CZCE', ts_code='SA2209.ZCE',start_date='20220101',end_date='20220630')

df1=pro.fut_basic(exchange='CZCE', fut_type='2')
df2=pro.fut_mapping(ts_code='SAL.ZCE')
df3=pro.fut_wsr(symbol='FG',)
print(df3)
data1=pd.DataFrame(df3)
#print(df)
data1.to_csv('E:\\test.csv',encoding='GB2312')
'''
#ts.set_token('4e7497b0629e1bf909b3efc087eadf39ef7002f9f4d57aea9a9303b0')
#pro=ts.pro_api()
#df=pro.fut_basic(exchange='CZCE', fut_type='2', fields='ts_code,symbol,name')
#df=pro.fut_daily(ts_code='SA2409.ZCE',start_date='20230915',end_date='20240805',fields=
#                     'ts_code,pre_close,pre_settle,trade_date,open,high,low,close')
#print(df)
class AverPriceClass():
    def __init__(self,token:str,
                 conn_user: str,
                 conn_pass: str,
                 conn_db: str,):
        ts.set_token(token=token)
        self._datadate = datetime.datetime.today().strftime("%Y%m%d")
        self._ts_pro=ts.pro_api()
        self._conn = cx_Oracle.connect(conn_user, conn_pass, conn_db)
        self._conn_cursor = self._conn.cursor()
        #self._pdconnect=create_engine('oracle://'+conn_user+":"+conn_pass+'@127.0.0.1:1521/orclpdb')
        print("------获取均价初始化开始-------")

    def _db_insert(self, sqlstr: str):
        self._conn_cursor.execute(sqlstr)
        self._conn.commit()
        print("[" + sqlstr + "]" + "写入数据库成功")

    def _db_update(self, sqlstr: str):
        self._conn_cursor.execute(sqlstr)
        self._conn.commit()
        print("[" + sqlstr + "]" + "更新数据库成功")

    def _db_select_rows(self, sqlstr: str) -> dict:
        ret_dict = {}
        self._conn_cursor.execute(sqlstr)
        columns = [col[0] for col in self._conn_cursor.description]
        rows = self._conn_cursor.fetchall()
        ret_dict['col_name'] = columns
        ret_dict['rows'] = rows
        self._conn_cursor.close()
        return ret_dict
    def _db_select_rows_list(self,sqlstr:str)->list:
        self._conn_cursor.execute(sqlstr)
        columns = [col[0] for col in self._conn_cursor.description]
        rows = self._conn_cursor.fetchall()
        result_list = [dict(zip(columns, row)) for row in rows]
        #self._conn_cursor.close()
        return result_list
    def _db_select_cnt(self, sqlstr: str):
        self._conn_cursor.execute(sqlstr)
        rows = self._conn_cursor.fetchall()
        return rows[0][0]

    def _get_list_bycolname(self,retlist:list,colname:str)->list:
        paralist = []
        for item in retlist:
            paralist.append(item.get(colname))
        return paralist

    def GetDailyDataByTsCode(self, Ts_code, Start_Date, End_Date):
        df = self._ts_pro.fut_daily(ts_code=Ts_code, start_date=Start_Date, end_date=End_Date, fields=
        'ts_code,trade_date,pre_close,pre_settle,open,high,low,close,settle,vol,oi,oi_chg')
        return df

    def GetTsCodeDF(self,flag:int):
        Future_Type_List_DF_CZCE = self._ts_pro.fut_basic(exchange='CZCE', fut_type='2', fields='ts_code,symbol,name')
        Future_Type_List_DF_DCE = self._ts_pro.fut_basic(exchange='DCE', fut_type='2', fields='ts_code,symbol,name')
        Future_Type_List_DF = self._ts_pro.concat([Future_Type_List_DF_CZCE, Future_Type_List_DF_DCE])
        Future_Type_List_MA = Future_Type_List_DF.loc[Future_Type_List_DF['symbol'].str[-1:] != 'L']
        Future_Type_List_LX = Future_Type_List_DF.loc[Future_Type_List_DF['symbol'].str[-1:] == 'L']
        if(flag==0):
            return Future_Type_List_MA
        else:
            return Future_Type_List_LX

    def GetTsCodeDFByDetail(self,instrumentidlist:[],symbollist:[],instrumentnamelist:[],exchangeidlist:[]):
        Future_Type_List_DF=pd.DataFrame({'ts_code':instrumentnamelist,'symbol':symbollist,'name':instrumentnamelist,'exchangeid':exchangeidlist})
        #Future_Type_List_DF.insert(0,'ts_code',instrumentid)
        #Future_Type_List_DF.insert(1,'symbol',instrumentid.split('.')[0])
        #Future_Type_List_DF.insert(2,'name',instrumentname)
        return Future_Type_List_DF

    def GetALLDataFrame(self, IniDF, Type_Df, Start_Date, End_Date):
        #for ts_code in zip(Type_Df['ts_code']):
        #    df = self.GetDailyDataByTsCode(ts_code[0], Start_Date, End_Date)
        #    IniDF = pd.concat([IniDF, df])
        IniDF.insert(loc=1,column='exchangeid',value=[Type_Df['exchangeid'][0]]*len(IniDF))
        IniDF.insert(loc=2,column='instrumentid',value=[Type_Df['symbol'][0]]*len(IniDF))
        IniDF.insert(loc=3,column='uptdate',value=[self._datadate]*len(IniDF))
        IniDF['pre_close'] = IniDF['pre_close'].astype('float32')
        IniDF['pre_settle'] = IniDF['pre_settle'].astype('float32')
        IniDF['open'] = IniDF['open'].astype('float32')
        IniDF['close'] = IniDF['close'].astype('float32')
        IniDF['high'] = IniDF['high'].astype('float32')
        IniDF['low'] = IniDF['low'].astype('float32')
        IniDF['vol'] = IniDF['vol'].astype('float32')
        IniDF['settle'] = IniDF['settle'].astype('float32')
        return IniDF

    def DF_Iint(self,df):
        df.insert(loc=len(df.columns), column='dateindex', value=df['trade_date'])  # 增加一列日期列
        x = df.copy()
        x.loc[:, 'trade_date'] = pd.to_datetime(x.loc[:, 'trade_date'])  # 将数据类型转换为日期类型,可以直接获取相应年份的数据
        # x = x.set_index('trade_date')
        df = x
        # df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df.rename(columns={'vol': 'volume'}, inplace=True)
        df = df.iloc[::-1]

        df.insert(loc=len(df.columns), column='MA5', value=df['close'].rolling(5).mean())
        df.insert(loc=len(df.columns), column='MA10', value=df['close'].rolling(10).mean())
        df.insert(loc=len(df.columns), column='MA20', value=df['close'].rolling(20).mean())
        df.insert(loc=len(df.columns), column='MA30', value=df['close'].rolling(30).mean())
        df.insert(loc=len(df.columns), column='MA60', value=df['close'].rolling(60).mean())
        return df

    def GetAverPriceDF(self,instrumentinfolist:dict):
        tscodelist=[instrumentinfolist.get('TS_CODE')]
        symbollist=[instrumentinfolist.get('TS_SYMBOL')]
        instrumentnamelist=[instrumentinfolist.get('STD_INSTRUMENTNAME')]
        exchangeidlist=[instrumentinfolist.get('TS_EXCHANGEID')]
        startdate=instrumentinfolist.get('CREATE_DATE')
        enddate=self._datadate
        tscode_df=self.GetTsCodeDFByDetail(instrumentidlist=tscodelist,symbollist=symbollist,
                                           instrumentnamelist=instrumentnamelist,exchangeidlist=exchangeidlist)
        tscode=tscodelist[0]
        data_df=self.GetDailyDataByTsCode(Ts_code=tscode,Start_Date=startdate,End_Date=enddate)
        final_df=self.GetALLDataFrame(IniDF=data_df,Type_Df=tscode_df,Start_Date=startdate,End_Date=enddate)
        ave_df=self.DF_Iint(final_df)
        dbdf=ave_df[['ts_code','exchangeid','instrumentid','MA5','MA10',
                      'MA20','MA30','MA60','dateindex','uptdate']]

        return dbdf

    def _df2db_insert(self,p_table_name, p_dataframe):
        # 准备数据，将待插入的dataframe转成list,以便多值插入
        tmp_list = np.array(p_dataframe).tolist()
        # 准备sql窜 形如 insert into table_name(a,b,c) values(:a,:b,:c)
        sql_string = 'insert into {}({}) values({})'.format(p_table_name, ','.join(list(p_dataframe.columns)),
                                                            ','.join(list(map(lambda x: ':' + x, p_dataframe.columns))))
        # 准备数据库连接，并插入数据库
        try:
            with self._conn as conn:
                with self._conn_cursor as cursor:
                    cursor.executemany(sql_string, tmp_list)
                    conn.commit()
        except cx_Oracle.Error as error:
            print('Error occurred:')
            print(error)
    def AverPriceDFToDB(self,ave_df:pd.DataFrame):
        av=""
    def test(self):
        instrumnetlist=['SA2409.ZCE','FG2409.ZCE','UR2409.ZCE','P2409.DCE']
        tscode_df=self.GetTsCodeDFByDetail(instrumentidlist=["SA2409.ZCE"],symbollist=["SA409"],instrumentnamelist=["纯碱2409"])
        start_date='20230915'
        end_date='20240807'
        ts_code="SA.ZCE"
        data_df=self.GetDailyDataByTsCode(Ts_code=ts_code,Start_Date=start_date,End_Date=end_date)
        f_df=self.GetALLDataFrame(data_df,tscode_df,start_date,end_date)
        ave_df=self.DF_Iint(f_df)
        print(ave_df)
    def test1(self):
        sql="select * from QUANT_FUTURE_MA_INSTRUMNET"
        ret_list=self._db_select_rows_list(sqlstr=sql)
        #first_dict=ret_dict.popitem()
        #ts_code_list=[ret_dict[k] for k in ['rows'] if k in ret_dict ]
        #ts_code_list= [ret_dict['rows'][v] for v in [ret_dict['col_name'].index(k) for k in ret_dict['col_name']]]
        #ts_code_list=[ret_dict['col_name'].index(k) for k in ret_dict['col_name']]
        #ts_code_list=ret_dict['rows']
        #ts_code_array=ret_dict[0]
        #ret_dict['rows'][0,[ret_dict['col_name'].index('SYMBOL')]]
        #item[retdict['col_name'].index('INSTRUMENTID')]
        ts_code_list=self._get_list_bycolname(retlist=ret_list,colname='TS_CODE')
        #print(ret_dict['rows'][0][ret_dict['col_name'].index('TS_CODE')])
        #print(ts_code_list[5])
    def test2(self):
        sql = "select * from QUANT_FUTURE_MA_INSTRUMNET"
        ret_list = self._db_select_rows_list(sqlstr=sql)
        dbdf=self.GetAverPriceDF(ret_list[0])
        #df=pd.read_sql(sql,self._pdconnect)
        self._df2db_insert('QUANT_FUTURE_AVG_PRICE',dbdf)
        #print(df)
if __name__=="__main__":
    #print('ddddddddd')
    token='4e7497b0629e1bf909b3efc087eadf39ef7002f9f4d57aea9a9303b0'
    averpriceCLS=AverPriceClass(token=token,conn_user="user_ph",conn_pass="ph",conn_db="127.0.1.1:1521/orclpdb")
    averpriceCLS.test2()