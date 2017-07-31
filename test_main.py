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
from CAlpha import Alphas

train_data = pd.read_csv("RB1710_train_min.csv")

train_data["vwap"] = ( train_data["turnover"].diff() / train_data["Volume"].diff() ).fillna(0)

train_data_panel = pd.Panel(data=train_data.values.reshape(1, train_data.shape[0], train_data.shape[1]),
                            items=["RB1709"], major_axis=train_data.index.tolist(), minor_axis=train_data.columns)

d = Alphas(train_data_panel)

def corr_cal(a1, a2):
    return pd.DataFrame(np.hstack([a1.values, a2.values]), columns=["var1", "var2"]).corr()

print(corr_cal( d.alpha001(), d.returns.shift(-1)))
print(corr_cal(d.alpha002(), d.returns.shift(-1)))
print(corr_cal(d.alpha003(), d.returns.shift(-1)))
print(corr_cal(d.alpha004(), d.returns.shift(-1)))
print(corr_cal(d.alpha005(), d.returns.shift(-1)))
print(corr_cal(d.alpha006(), d.returns.shift(-1)))
print(corr_cal(d.alpha007(), d.returns.shift(-1)))
print(corr_cal(d.alpha008(), d.returns.shift(-1)))
print(corr_cal(d.alpha009(), d.returns.shift(-1)))
print(corr_cal(d.alpha010(), d.returns.shift(-1)))
print(corr_cal(d.alpha011(), d.returns.shift(-1)))
print(corr_cal(d.alpha012(), d.returns.shift(-1)))
print(corr_cal(d.alpha013(), d.returns.shift(-1)))

train_data = np.column_stack((d.alpha001(), d.alpha002(),
                              d.alpha003(), d.alpha004(),
                              d.alpha005(), d.alpha006(),
                              d.alpha007(), d.alpha008(),
                              d.alpha010(), d.alpha009(),
                              d.alpha011(), d.alpha012(), d.returns))

from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=10)