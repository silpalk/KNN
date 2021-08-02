# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:51:33 2021

@author: Amarnadh Tadi
"""
import pandas as pd
glass=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign8\glass.csv")
glass.columns
glass.head()
glass.shape
x=glass.iloc[:,:9]
y=glass.iloc[:,[9]]

##splitting of data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)
##Generating Model for K=3
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
##fit the model on train data
model.fit(x_train,y_train)
## predicting of test values 
y_predict=model.predict(x_test)
## measuring of accuracy 
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)

##Generating Model for K=3
from sklearn.neighbors import KNeighborsClassifier
model1=KNeighborsClassifier(n_neighbors=5)
##fit the model on train data
model1.fit(x_train,y_train)
## predicting of test values 
y_predict1=model1.predict(x_test)
## measuring of accuracy 
from sklearn import metrics
accuracy1=metrics.accuracy_score(y_test,y_predict1)

##for zoo data set
import pandas as pd
zoo=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign8\Zoo.csv")
zoo.columns
zoo.dtypes
zoo.head()
zoo.shape
zoo.drop(['animal name'],axis=1,inplace=True)

##splitting data to features and target
x=zoo.iloc[:,:16]

y=zoo.iloc[:,[16]]
##splitting of data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.35)
##Generating Model for K=3
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=2)
##fit the model on train data
model.fit(x_train,y_train)
## predicting of test values 
y_predict=model.predict(x_test)
## measuring of accuracy 
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

##Generating Model for K=3
from sklearn.neighbors import KNeighborsClassifier
model1=KNeighborsClassifier(n_neighbors=5)
##fit the model on train data
model1.fit(x_train,y_train)
## predicting of test values 
y_predict1=model1.predict(x_test)
## measuring of accuracy 
from sklearn import metrics
accuracy1=metrics.accuracy_score(y_test,y_predict1)
accuracy1

