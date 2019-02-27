# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:26:35 2019

@author: pushparajkarthick_d
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv("D:\\karthick\\study\\ML\\MachineLearning\\try\\Admission_Predict_Ver1.1.csv")
#print(data.head())

new_data = data.drop('Serial No.',axis = 1)
#print(new_data.head())

#print(new_data.describe())

print(new_data.isnull().sum(axis=0))

f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(new_data.corr(),annot=True)
plt.draw()
plt.show(block=False)

"""plt.subplots(figsize=(20,4))
sns.barplot(x="GRE Score",y="Chance_of_Admit", data=data)
plt.subplots(figsize=(25,5))
sns.barplot(x="TOEFL Score",y="Chance of Admit", data=data)
plt.subplots(figsize=(20,4))
sns.barplot(x="University Rating",y="Chance of Admit", data=data)
plt.subplots(figsize=(15,5))
sns.barplot(x="SOP",y="Chance of Admit", data=data)
"""

"""temp_series = new_data.Research.value_counts()
labels = np.array(temp_series.index)
sizes = (np.array(temp_series/temp_series.sum())*100)
colors = ['Pink', 'Skyblue']
plt.pie(sixes,labels = labels,colors=colors
"""


X=new_data.iloc[:,:7]
y=new_data["Chance of Admit "]

print(X.shape)
print(y.shape)

print(X.head())

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=50)
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
print(y_pred)

from sklearn.metrics import mean_absolute_error, r2_score

print("R2 score ", r2_score(y_pred,y_test))
print("Mean_absolute_error" , mean_absolute_error(y_pred,y_test))


