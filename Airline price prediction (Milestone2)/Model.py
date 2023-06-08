import random
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from preprocess_1 import *
#########################################################
data = pd.read_csv('airline-price-classification.csv')

# There no null value
data.info()
# Features
X = data.iloc[:, :10]
# Label
Y = []

# Pre_Processing
# route is two things source and destination
route(X, 'route')
stop_fun(X, 'stop')
# read the text feature from the data
cols = ('ch_code', 'type', 'airline', 'source', 'destination', 'stop')
feature_encoder(X, cols)

# deal with date by splitting  it into day, Month and year
date_handel(X, 'date')


###############################################
# deal with time dep_time & arr_time
time_handel(X, 'dep_time')
time_handel(X, 'arr_time')
time_taken(X, 'arr_time', 'dep_time')

# {cheap = 0 , moderate = 1, expensive = 2 or very expensive = 3}
target = data['TicketCategory']
classify_TicketCategory(Y, target)

X.info()
################################################
#logistic regression

# xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3,random_state=0,shuffle=True)
#
# start_time=time.time()
# model=LogisticRegression(solver='liblinear', C=0.08,multi_class='ovr',random_state=1)
#
# model.fit(xtrain,ytrain)
# y_pred=model.predict(xtest)
# endtime=time.time()
# print('time is :',endtime-start_time )
#
# # print(accuracy_score(ytest, y_pr))
# print(y_pred)
# print ("Accuracy : ", accuracy_score(ytest, y_pred))

# #==============================================================================

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier
X=feature_scaling(X)


xtrain_s,xtest_s,ytrain_s,ytest_s=train_test_split(X,Y,test_size=0.3,random_state=1,shuffle=True)
start_time =time.time()
classifier  = RandomForestClassifier(n_estimators = 12)
classifier.fit(xtrain_s,ytrain_s)


y_p=classifier.predict(xtest_s)
print(y_p)
print ("Accuracy : ", accuracy_score(ytest_s, y_p))




endtime=time.time()
time=endtime-start_time
print('time is :',time )


# #===================================================================================
#
#
# xtrain_t,xtest_t,ytrain_t,ytest_t=train_test_split(X,Y,test_size=0.3,shuffle=True)
# # svc=SVC(probability=True, kernel='poly',degree=4)
# starttime=time.time()
# from sklearn.ensemble import AdaBoostClassifier
# clf_model_dtree=AdaBoostClassifier(n_estimators=12,learning_rate=1)
#
# clf_model_dtree=clf_model_dtree.fit(xtrain_t,ytrain_t)
# endtime =time.time()
# time=endtime-starttime
# print('time is :',time)
# y_pred_dtree=clf_model_dtree.predict(xtest_t)
# print ("Accuracy : ", accuracy_score(ytest_t, y_pred_dtree))
