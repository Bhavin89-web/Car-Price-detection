#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:55:37 2020

@author: bhavin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(rc={'figure.figsize':(11.7,8.27)})

cars_data=pd.read_csv("cars_sampled.csv")
cars_data.columns

cars_data['name'].unique()
cars_data['seller'].unique()
cars_data['offerType'].unique()
cars_data['price'].unique()
cars_data['fuelType'].unique()
cars_data['vehicleType'].unique()
cars_data['gearbox'].unique()
cars_data['powerPS'].unique()
cars_data['model'].unique()


cars=cars_data.copy()

cars.info()

cars.describe()

pd.set_option('display.float_format',lambda x:'%.3f' %x)
cars.describe()

pd.set_option('display.max_columns',50)
cars.describe()

#drop unwanted info

col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col,axis=1)

##remove dupliactes
cars.drop_duplicates(keep='first',inplace=True)

#470 duplicates

#null values
cars.isnull().sum()


##year of registration
yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']> 2018)
sum(cars['yearOfRegistration']< 1950)

sns.regplot(x='yearOfRegistration',y='price',scatter=True,
            fit_reg=False,data=cars)
#working range 1950 and 2018

#variable price
price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)

#working range 100 t0 15000 and skewness is there in data
#variable powerps
power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,
            fit_reg=False,data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)

#working range 10 and 500


#data cleaning : working range of data


cars=cars[(cars.yearOfRegistration<=2018)
         &(cars.yearOfRegistration>=1950)
         &(cars.price>=100)
         &(cars.price<=150000)
         &(cars.powerPS>=10)
         &(cars.powerPS<=500)]

##6700 record deleted

#combining year of registration and monthof registration
cars['monthOfRegistration']/=12

#combine new variable age by adding yearof registration and month
cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()


#drop year and month of registration
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#visualize parameters
#age
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

#price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#powerps
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

##Visualizing parameters after narrowing
#age vs price
sns.regplot(x='Age',y='price',scatter=True,
            fit_reg=False,data=cars)

###cars price higher are newer
#with increase age price decrease
#however same cars are priced higher with increase in age

#powerps vs price
sns.regplot(x='powerPS',y='price',scatter=True,
            fit_reg=False,data=cars)



#variable seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)
sns.countplot(x='seller',data=cars)
#fewer cars have 'commercial'=>insignificant

##variablr offertype
cars['offerType'].value_counts()
sns.countplot(x='offerType',data=cars)
##all cars have offer=> insignificant


#variable abtest
cars['abtest'].value_counts()

pd.crosstab(cars['abtest'],columns='count',normalize=True)

sns.countplot(x='abtest',data=cars)
#equally distributed

sns.boxplot(x='abtest',y='price',data=cars)

#for every price value there is almost 50-50 
#distribution
#does not affect the price =>insignificant

#variable vehicle type
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)

#variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)

sns.countplot(x='gearbox',data=cars)
sns.boxplot(x='gearbox',y='price',data=cars)

#gearbox column affect the price

#variable model
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
plt.figure(figsize = (60,6))
sns.countplot(x='model',data=cars)
plt.figure(figsize = (60,6))
sns.boxplot(x='model',y='price',data=cars)
#cars are distributed over many models
#considering in modelling

#variable kilometer
cars['kilometer'].value_counts()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
cars['kilometer'].describe()
sns.countplot(x='kilometer',data=cars)
sns.boxplot(x='kilometer',y='price',data=cars)

#considering modeling

#varible fueltype

cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
cars['fuelType'].describe()
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)

#considering modeling
#variable brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
cars['brand'].describe()
sns.countplot(x='brand',data=cars)
sns.boxplot(x='brand',y='price',data=cars)
#considering modeling

#notRepairedDamage

cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
sns.boxplot(x='notRepairedDamage',y='price',data=cars)
#considering modeling


#age

cars['Age'].value_counts()
pd.crosstab(cars['Age'],columns='count',normalize=True)
sns.countplot(x='Age',data=cars)
sns.boxplot(x='Age',y='price',data=cars)
#considering modeling

#removing insignificant variable

col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()

#correlation

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

#ommmiting missing rows
cars_omit=cars_copy.dropna(axis=0)

##converting categorical variables to numeric
cars_omit=pd.get_dummies(cars_omit,drop_first=True)

######

#importing necessary models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

x1=cars_omit.drop(['price'],axis=1)
y1=cars_omit['price']

#plotting variable price
prices=pd.DataFrame({"1. Before": y1, "2.After ":np.log(y1)})
prices.hist()

y1=np.log(y1)

#split
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)


##making base model by using data mean value
#this is to set benchmark and compare our regression model

base_pred=np.mean(y_test)
print(base_pred)


##repeat same value till length of test_data
base_pred=np.repeat(base_pred,len(y_test))

##finding RMSE
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))

print(base_root_mean_square_error)


#####linear Regression
l_regressor=LinearRegression(fit_intercept=True)
model_lin1=l_regressor.fit(x_train,y_train)

y_pred1=l_regressor.predict(x_test)

lin_mse1=mean_squared_error(y_test,y_pred1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

#r squared value
r2_lin_test1= model_lin1.score(x_test,y_test)
r2_lin_train1= model_lin1.score(x_train,y_train)

######
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
l_model=regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
#test score and train score

test1=l_model.score(x_test,y_test)
train1=l_model.score(x_train,y_train)

from sklearn import metrics
print(metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


####Residual plot analysis
residual1=y_test-y_pred
sns.regplot(x=y_pred,y=residual1,scatter=True,
            fit_reg=False,data=cars)
#########

residual1.describe()

####random Forest

rf=RandomForestRegressor(n_estimators=100)
model_rf=rf.fit(x_train,y_train)

y_pred2=rf.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred2))
#test score and train score

rf_test1=model_rf.score(x_test,y_test)
rf_train1=model_rf.score(x_train,y_train)

from sklearn import metrics
print(metrics.mean_squared_error(y_test,y_pred2))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred2)))


##########
from sklearn.svm import SVR
regressor5=SVR()
regressor5.fit(x_train,y_train)
y_pred4=regressor5.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred4))

from sklearn import metrics
print(metrics.mean_squared_error(y_test,y_pred4))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred4)))
metrics.mean_squared_log_error(y_test,y_pred4)
