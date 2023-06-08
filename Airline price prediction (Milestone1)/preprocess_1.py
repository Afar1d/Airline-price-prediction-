import datetime as dt

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def feature_encoder(x, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))
    return x


def route(x, cols):
    x['source'] = x[cols].str.split('\'', expand=True)[3]
    x['destination'] = x[cols].str.split('\'', expand=True)[7]
    x.drop(columns=[cols], inplace=True)


def date_handel(d, cols):
    d['date_year'] = d[cols].str.split('-|/', expand=True)[2].astype(np.int16)
    d['date_month'] = d[cols].str.split('-|/', expand=True)[1].astype(np.int8)
    d['date_day'] = d[cols].str.split('-|/', expand=True)[0].astype(np.int8)
    d.drop(columns=[cols], inplace=True)
    # we can also Extract Month Name, Day of Week-Name ,  Extract Day of Week


def stop_fun(x, cols):
    x[cols] = x[cols].str.split('p', expand=True)[0] + 'p'


def time_handel(d2, cols2):

    # converting to datatime datatype
    d2[cols2] = pd.to_datetime(d2[cols2])
    # all dep_time in minute
    d2[cols2] = d2[cols2].dt.minute + d2[cols2].dt.hour*100


def time_taken(d4, time1, time2):
    d4['time_taken'] = abs(d4[time1]-d4[time2])




def feature_scaling(x):
    """ Standardisation by AMF"""
    standardisation = preprocessing.StandardScaler()
    # Scaled feature
    x_after_standardisation = standardisation.fit_transform(x)
    return x_after_standardisation


