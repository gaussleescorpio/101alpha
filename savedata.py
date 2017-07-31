from __future__ import division, print_function, absolute_import
import glob
from imp import reload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns=999
import tflearn
import tensorflow as tf
from tflearn.data_utils import to_categorical, pad_sequences
import helper

instrument_list = pd.read_csv("InstrumentInfo.txt")
exc = "SHFE_10"
data_folder = "/run/user/1001/gvfs/smb-share:server=172.30.50.120,share=nautilusdata"
future_list = ["RB1710"]

data_dict = {}
test_data_dict = {}
for ticker in future_list:
    print("Trying to get the price for %s" % ticker)
    df = helper.getOverDayData2(data_folder, exc, ticker,
                                None, startDate=20170617,
                                endDate=20170705, drop_na=False)
    df.reset_index(inplace=True)
    df = df.loc[~df.index.duplicated(keep='first')]
    df = df.sort_values(by="utcReceiveTime")
    data_dict[ticker] = df

df = None
for ticker in future_list:
    print("Trying to get the price for %s" % ticker)
    df = helper.getOverDayData2(data_folder, exc, ticker,
                                None, startDate=20170706,
                                endDate=20170714, drop_na=False)
    df.reset_index(inplace=True)
    df = df.loc[~df.index.duplicated(keep='first')]
    df = df.sort_values(by="utcReceiveTime")
    test_data_dict[ticker] = df

df = data_dict["RB1710"]
test_df = test_data_dict["RB1710"]
df = helper.getBar(df, 120)
test_df = helper.getBar(test_df, 120)

df.to_csv("RB1710_train_min.csv")
test_df.to_csv("RB1710_test_min.csv")


